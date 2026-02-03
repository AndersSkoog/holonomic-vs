#version 330 core
layout (location = 0) in vec3 mesh_vertex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 mesh_color;

out VsOut {
    vec3 pos;
    vec3 col;
} vs_out;

void main()
{
    vec4 worldPos = uModel * vec4(mesh_vertex, 1.0);
    vs_out.pos = worldPos.xyz;
    vs_out.col = mesh_color;
    gl_Position = uProjection * uView * worldPos;
}
