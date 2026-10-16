mod math;
mod render;

use math::{Mat4, Vec3};
use render::{create_context, create_pipeline, RenderContext};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

thread_local! {
    static RENDERER: RefCell<Option<Renderer>> = RefCell::new(None);
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

struct Renderer {
    context: RenderContext,
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    meshes: Vec<Mesh>,
    clear_color: wgpu::Color,
    view_projection: Mat4,
}

impl Renderer {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let context = create_context(canvas).await?;

        let uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("uniform_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let uniform_bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline = create_pipeline(&context, SHADER, &uniform_bind_group_layout);
        let depth_view = create_depth(&context);

        Ok(Renderer {
            context,
            pipeline,
            depth_view,
            uniform_buffer,
            uniform_bind_group,
            meshes: Vec::new(),
            clear_color: wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            view_projection: Mat4::IDENTITY,
        })
    }

    fn create_mesh(&mut self, vertices: &[f32], indices: &[u16]) -> u32 {
        let vertex_buffer = create_buffer(
            &self.context.device,
            "vertices",
            bytemuck::cast_slice(vertices),
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = create_buffer(
            &self.context.device,
            "indices",
            bytemuck::cast_slice(indices),
            wgpu::BufferUsages::INDEX,
        );

        let mesh = Mesh {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        };

        self.meshes.push(mesh);
        (self.meshes.len() - 1) as u32
    }

    fn set_clear_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.clear_color = wgpu::Color {
            r: r as f64, g: g as f64, b: b as f64, a: a as f64,
        };
    }

    fn set_view_projection(&mut self, mvp: &[f32]) {
        if mvp.len() == 16 {
            self.view_projection = Mat4::from_cols_array(mvp.try_into().unwrap_or(&[0.0; 16]));
        }
    }

    fn draw_mesh(&mut self, id: u32, model: &[f32]) {
        let mesh = match self.meshes.get(id as usize) {
            Some(m) => m,
            None => return,
        };

        let model_mat = if model.len() == 16 {
            Mat4::from_cols_array(model.try_into().unwrap_or(&[0.0; 16]))
        } else {
            Mat4::IDENTITY
        };

        let mvp = self.view_projection * model_mat;
        let uniforms = Uniforms { mvp: mvp.to_cols_array_2d() };
        self.context.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let frame = match self.context.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.context.surface.configure(&self.context.device, &self.context.config);
                return;
            }
        };

        let view_texture = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_texture,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }

        self.context.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.context.config.width = width;
        self.context.config.height = height;
        self.context.surface.configure(&self.context.device, &self.context.config);
        self.depth_view = create_depth(&self.context);
    }
}

fn with_renderer<F: FnOnce(&mut Renderer)>(f: F) {
    RENDERER.with(|r| {
        if let Some(renderer) = r.borrow_mut().as_mut() {
            f(renderer);
        }
    });
}

#[wasm_bindgen]
pub fn js_create_mesh(vertices: &[f32], indices: &[u16]) -> u32 {
    let mut id = 0;
    with_renderer(|r| { id = r.create_mesh(vertices, indices); });
    id
}

#[wasm_bindgen]
pub fn js_set_clear_color(r: f32, g: f32, b: f32, a: f32) {
    with_renderer(|rr| rr.set_clear_color(r, g, b, a));
}

#[wasm_bindgen]
pub fn js_set_view_projection(mvp: &[f32]) {
    with_renderer(|r| r.set_view_projection(mvp));
}

#[wasm_bindgen]
pub fn js_draw_mesh(id: u32, model: &[f32]) {
    with_renderer(|r| r.draw_mesh(id, model));
}

#[wasm_bindgen]
pub fn js_resize(width: u32, height: u32) {
    with_renderer(|r| r.resize(width, height));
}

#[wasm_bindgen]
pub fn make_perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Vec<f32> {
    Mat4::perspective_rh(fov, aspect, near, far).to_cols_array().to_vec()
}

#[wasm_bindgen]
pub fn make_look_at(ex: f32, ey: f32, ez: f32, tx: f32, ty: f32, tz: f32) -> Vec<f32> {
    Mat4::look_at_rh(Vec3::new(ex, ey, ez), Vec3::new(tx, ty, tz), Vec3::Y)
        .to_cols_array().to_vec()
}

#[wasm_bindgen]
pub fn make_translation(x: f32, y: f32, z: f32) -> Vec<f32> {
    Mat4::from_translation(Vec3::new(x, y, z)).to_cols_array().to_vec()
}

#[wasm_bindgen]
pub fn make_rotation(x: f32, y: f32, z: f32) -> Vec<f32> {
    Mat4::from_euler(glam::EulerRot::XYZ, x, y, z).to_cols_array().to_vec()
}

#[wasm_bindgen]
pub fn make_scale(x: f32, y: f32, z: f32) -> Vec<f32> {
    Mat4::from_scale(Vec3::new(x, y, z)).to_cols_array().to_vec()
}

#[wasm_bindgen]
pub fn multiply_matrices(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.len() != 16 || b.len() != 16 { return vec![0.0; 16]; }
    let ma = Mat4::from_cols_array(a.try_into().unwrap());
    let mb = Mat4::from_cols_array(b.try_into().unwrap());
    (ma * mb).to_cols_array().to_vec()
}

#[no_mangle]
pub extern "C" fn as_create_mesh(vertices_ptr: *const f32, vertices_len: usize, indices_ptr: *const u16, indices_len: usize) -> u32 {
    let vertices = unsafe { std::slice::from_raw_parts(vertices_ptr, vertices_len) };
    let indices = unsafe { std::slice::from_raw_parts(indices_ptr, indices_len) };
    let mut id = 0;
    with_renderer(|r| { id = r.create_mesh(vertices, indices); });
    id
}

#[no_mangle]
pub extern "C" fn as_set_clear_color(r: f32, g: f32, b: f32, a: f32) {
    with_renderer(|rr| rr.set_clear_color(r, g, b, a));
}

#[no_mangle]
pub extern "C" fn as_set_view_projection(ptr: *const f32) {
    let mvp = unsafe { std::slice::from_raw_parts(ptr, 16) };
    with_renderer(|r| r.set_view_projection(mvp));
}

#[no_mangle]
pub extern "C" fn as_draw_mesh(id: u32, ptr: *const f32) {
    let model = unsafe { std::slice::from_raw_parts(ptr, 16) };
    with_renderer(|r| r.draw_mesh(id, model));
}

#[no_mangle]
pub extern "C" fn as_resize(width: u32, height: u32) {
    with_renderer(|r| r.resize(width, height));
}

#[no_mangle]
pub extern "C" fn as_make_perspective(fov: f32, aspect: f32, near: f32, far: f32, out_ptr: *mut f32) {
    let m = Mat4::perspective_rh(fov, aspect, near, far);
    unsafe { std::ptr::copy_nonoverlapping(m.to_cols_array().as_ptr(), out_ptr, 16); }
}

#[no_mangle]
pub extern "C" fn as_make_look_at(ex: f32, ey: f32, ez: f32, tx: f32, ty: f32, tz: f32, out_ptr: *mut f32) {
    let m = Mat4::look_at_rh(Vec3::new(ex, ey, ez), Vec3::new(tx, ty, tz), Vec3::Y);
    unsafe { std::ptr::copy_nonoverlapping(m.to_cols_array().as_ptr(), out_ptr, 16); }
}

#[no_mangle]
pub extern "C" fn as_make_rotation(x: f32, y: f32, z: f32, out_ptr: *mut f32) {
    let m = Mat4::from_euler(glam::EulerRot::XYZ, x, y, z);
    unsafe { std::ptr::copy_nonoverlapping(m.to_cols_array().as_ptr(), out_ptr, 16); }
}

#[no_mangle]
pub extern "C" fn as_multiply_matrices(a_ptr: *const f32, b_ptr: *const f32, out_ptr: *mut f32) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, 16) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, 16) };
    let ma = Mat4::from_cols_array(a.try_into().unwrap());
    let mb = Mat4::from_cols_array(b.try_into().unwrap());
    let result = (ma * mb).to_cols_array();
    unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, 16); }
}

#[wasm_bindgen]
pub async fn init_renderer() -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;

    let rect = canvas.get_bounding_client_rect();
    canvas.set_width(rect.width() as u32);
    canvas.set_height(rect.height() as u32);

    let renderer = Renderer::new(canvas).await?;
    RENDERER.with(|r| { *r.borrow_mut() = Some(renderer); });

    let on_resize = Closure::<dyn FnMut()>::new(move || {
        let w = web_sys::window().unwrap();
        let doc = w.document().unwrap();
        let canvas = doc.get_element_by_id("canvas").unwrap();
        let canvas: HtmlCanvasElement = canvas.dyn_into().unwrap();
        let rect = canvas.get_bounding_client_rect();
        canvas.set_width(rect.width() as u32);
        canvas.set_height(rect.height() as u32);
        with_renderer(|r| r.resize(rect.width() as u32, rect.height() as u32));
    });

    window
        .add_event_listener_with_callback("resize", on_resize.as_ref().unchecked_ref())
        .unwrap();
    on_resize.forget();

    Ok(())
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}

fn create_buffer(device: &wgpu::Device, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

fn create_depth(context: &RenderContext) -> wgpu::TextureView {
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: context.config.width,
            height: context.config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

const SHADER: &str = r#"
struct Uniforms { mvp: mat4x4<f32>, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};
@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;
