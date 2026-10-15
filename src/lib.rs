mod math;
mod render;

use math::{Mat4, Vec3};
use render::{create_context, create_pipeline, RenderContext};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    position: Vec3,
    rotation: Vec3,
}

#[wasm_bindgen]
pub struct Renderer {
    context: RenderContext,
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    meshes: Vec<Mesh>,
}

#[wasm_bindgen]
impl Renderer {
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Renderer, JsValue> {
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
        })
    }

    pub fn create_mesh(&mut self, vertices: &[f32], indices: &[u16]) -> u32 {
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
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
        };

        self.meshes.push(mesh);
        (self.meshes.len() - 1) as u32
    }

    pub fn set_position(&mut self, id: u32, x: f32, y: f32, z: f32) {
        if let Some(mesh) = self.meshes.get_mut(id as usize) {
            mesh.position = Vec3::new(x, y, z);
        }
    }

    pub fn set_rotation(&mut self, id: u32, x: f32, y: f32, z: f32) {
        if let Some(mesh) = self.meshes.get_mut(id as usize) {
            mesh.rotation = Vec3::new(x, y, z);
        }
    }

    pub fn render(&mut self) {
        let aspect = self.context.config.width as f32 / self.context.config.height as f32;

        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);

        let frame = match self.context.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.context
                    .surface
                    .configure(&self.context.device, &self.context.config);
                return;
            }
        };

        let view_texture = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_texture,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.1,
                            a: 1.0,
                        }),
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

            for mesh in &self.meshes {
                let model = Mat4::from_translation(mesh.position)
                    * Mat4::from_euler(
                        glam::EulerRot::XYZ,
                        mesh.rotation.x,
                        mesh.rotation.y,
                        mesh.rotation.z,
                    );
                let mvp = proj * view * model;

                let uniforms = Uniforms {
                    mvp: mvp.to_cols_array_2d(),
                };
                self.context
                    .queue
                    .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }

        self.context.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.context.config.width = width;
        self.context.config.height = height;
        self.context
            .surface
            .configure(&self.context.device, &self.context.config);
        self.depth_view = create_depth(&self.context);
    }
}

fn create_buffer(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
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
struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOutput {
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

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;

    let width = window.inner_width().unwrap().as_f64().unwrap() as u32;
    let height = window.inner_height().unwrap().as_f64().unwrap() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let renderer = Rc::new(RefCell::new(Renderer::new(canvas).await?));

    let (vertices, indices) = cube_data();
    renderer.borrow_mut().create_mesh(&vertices, &indices);

    let on_resize = {
        let renderer = renderer.clone();
        let closure = Closure::<dyn FnMut()>::new(move || {
            let w = web_sys::window().unwrap();
            let width = w.inner_width().unwrap().as_f64().unwrap() as u32;
            let height = w.inner_height().unwrap().as_f64().unwrap() as u32;
            renderer.borrow_mut().resize(width, height);
        });
        closure
    };

    window
        .add_event_listener_with_callback("resize", on_resize.as_ref().unchecked_ref())
        .unwrap();
    on_resize.forget();

    let last = Rc::new(RefCell::new(window.performance().unwrap().now()));
    schedule_frame(renderer, last, window);
    Ok(())
}

fn schedule_frame(
    renderer: Rc<RefCell<Renderer>>,
    last: Rc<RefCell<f64>>,
    window: web_sys::Window,
) {
    let r = renderer.clone();
    let l = last.clone();
    let w = window.clone();

    let closure = Closure::once(move || {
        let now = w.performance().unwrap().now();
        let _dt = ((now - *l.borrow()) / 1000.0) as f32;
        *l.borrow_mut() = now;

        let mut renderer = r.borrow_mut();
        let rot = now as f32 * 0.001;
        renderer.set_rotation(0, 0.0, rot, 0.0);
        renderer.render();

        schedule_frame(r.clone(), l.clone(), w.clone());
    });

    window
        .request_animation_frame(closure.as_ref().unchecked_ref())
        .unwrap();
    closure.forget();
}

fn cube_data() -> (Vec<f32>, Vec<u16>) {
    let vertices: Vec<f32> = vec![
        -1.0, -1.0, -1.0,  0.0, 0.0, 1.0,
         1.0, -1.0, -1.0,  0.0, 1.0, 1.0,
         1.0,  1.0, -1.0,  0.0, 1.0, 0.0,
        -1.0,  1.0, -1.0,  1.0, 1.0, 0.0,
        -1.0, -1.0,  1.0,  1.0, 0.0, 0.0,
         1.0, -1.0,  1.0,  1.0, 0.0, 1.0,
         1.0,  1.0,  1.0,  1.0, 1.0, 1.0,
        -1.0,  1.0,  1.0,  0.0, 0.0, 0.0,
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2,  0, 2, 3,
        4, 6, 5,  4, 7, 6,
        0, 3, 7,  0, 7, 4,
        1, 5, 6,  1, 6, 2,
        0, 4, 5,  0, 5, 1,
        3, 2, 6,  3, 6, 7,
    ];

    (vertices, indices)
                        }
