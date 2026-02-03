#version 330 core
in vec3 vPos;
in vec3 vColor;

uniform float light_mul;
uniform float light_add;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vPos);
    float light = dot(n, normalize(vec3(1.0,1.0,1.0))) * light_mul + light_add;
    FragColor = vec4(vColor * light, 1.0);
}
