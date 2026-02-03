#version 330 core
layout (location = 0) in vec2 disc_point;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCol;
uniform vec2 sel_disc_point;

out VsOut {
    vec3 pos;
    vec3 col;
} vs_out;

float polar_angle(vec2 p){
    return atan(p.y, p.x);
}

vec2 rot2(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return vec2(c*p.x - s*p.y, s*p.x + c*p.y);
}

vec4 disc_to_sphere(vec2 p, float R){
    float r2 = dot(p, p);
    float d = r2 + R*R;
    return vec4(
        2.0 * R * p.x / d,
        2.0 * R * p.y / d,
        (r2 - R*R) / d,
        1.0
    );
}

void main()
{
    float a = polar_angle(sel_disc_point);
    vec2 rp = rot2(disc_point, a);
    vec4 worldPos = uModel * disc_to_sphere(rp, 1.0);
    vs_out.pos = worldPos.xyz;
    vs_out.col = uCol;
    gl_Position = uProjection * uView * worldPos;
}
