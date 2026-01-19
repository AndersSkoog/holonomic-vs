#version 330

//-------------------------constants--------------------------------------------------

#define EPS8  (1.0/float(1<<8))
#define EPS16 (1.0/float(1<<16))
#define EPS32 (1.0/float(1<<32))
#define PI 3.141592653589793
#define TAU (2.0 * PI)

//-----Vec utils-----------------------------------------------------------------------

vec3 vec3_normalize_safe(vec3 v){
    float l = length(v);
    if (abs(l) < EPS16) l = EPS16;
    return v / l;
}

vec2 vec2_normalize_safe(vec2 v){
    float l = length(v);
    if (abs(l) < EPS16) l = EPS16;
    return v / l;
}

//------Structs-------------------------------------------------------------------------

struct cplx {
    float re;
    float im;
};

struct cplx2 {
    cplx z1;
    cplx z2;
};

struct persp {
    vec3 frwd;
    vec3 back;
    vec3 right;
    vec3 left;
    vec3 up;
    vec3 down;
    vec3 pos;
    vec3 target;
};

struct sphere_pt {
    float x;
    float y;
    float z;
    float r;
    float theta;
    float phi;
};

struct sphere_orient_3 {
  vec3 center_coord;
  vec3 r3_cord;
  vec3 s2_cord;
  vec3 radial_dir;
  vec3 cube_verts[8];
  cplx slope;
  mat3 rot_mtx;
  mat3 xy_plane;
  mat3 xz_plane;
  mat3 yz_plane;
  cplx

};









// --- Constructors ------------------------------------------------------------

persp perspective(vec3 pos, vec3 target){
    float lim = 1.0 - EPS16;
    vec3 frwd = vec3_normalize_safe(target - pos);

    vec3 world_up =
        (abs(dot(frwd, vec3(0,0,1))) > lim) ?
        vec3(0,1,0) : vec3(0,0,1);

    vec3 right = vec3_normalize_safe(cross(frwd, world_up));
    vec3 up = cross(right, frwd);

    persp ret;
    ret.frwd = frwd;
    ret.back = -frwd;
    ret.right = right;
    ret.left = -right;
    ret.up = up;
    ret.down = -up;
    ret.pos = pos;
    ret.target = target;
    return ret;
}

cplx2 c2_make(cplx z1, cplx z2){
    cplx2 ret;
    ret.z1 = z1;
    ret.z2 = z2;
    return ret;
}

cplx c_make(float re, float im) {
    cplx z;
    z.re = re;
    z.im = im;
    return z;
}

cplx c_from_vec2(vec2 v) { return c_make(v.x, v.y); }
vec2 c_to_vec2(cplx z)   { return vec2(z.re, z.im); }

// --- Complex Arithmetic --------------------------------------------------------

cplx c_add(cplx a, cplx b){ return c_make(a.re+b.re, a.im+b.im); }
cplx c_sub(cplx a, cplx b){ return c_make(a.re-b.re, a.im-b.im); }

cplx c_mul(cplx a, cplx b){
    return c_make(a.re*b.re - a.im*b.im,
                  a.re*b.im + a.im*b.re);
}

cplx c_div(cplx a, cplx b){
    float d = b.re*b.re + b.im*b.im;
    return c_make((a.re*b.re + a.im*b.im)/d,
                  (a.im*b.re - a.re*b.im)/d);
}

cplx c_conj(cplx z){ return c_make(z.re, -z.im); }

// ---Complex Magnitude & Phase -------------------------------------------------------

float c_mag(cplx z){ return length(vec2(z.re,z.im)); }
float c_phase(cplx z){ return atan(z.im, z.re); }

// --- Polar / Rect ------------------------------------------------------------

vec2 c_polar(cplx z){ return vec2(c_mag(z), c_phase(z)); }
cplx c_rect(float r, float phi){ return c_make(r*cos(phi), r*sin(phi)); }

// --- Exponential / Logarithm / Power -----------------------------------------

cplx c_exp(cplx z){
    float ex = exp(z.re);
    return c_make(ex*cos(z.im), ex*sin(z.im));
}

cplx c_log(cplx z){
    return c_make(log(c_mag(z)), c_phase(z));
}

cplx c_pow(cplx z, cplx w){
    return c_exp(c_mul(c_log(z), w));
}

cplx c_scale(cplx z, float s){
    return c_make(z.re*s, z.im*s);
}

// --- Complex Trigonometric-----------------------------------------------------------

cplx c_sin(cplx z){
    return c_make( sin(z.re)*cosh(z.im),
                   cos(z.re)*sinh(z.im) );
}

cplx c_cos(cplx z){
    return c_make( cos(z.re)*cosh(z.im),
                  -sin(z.re)*sinh(z.im) );
}

cplx c_tan(cplx z){
    return c_div(c_sin(z), c_cos(z));
}

// --- Hyperbolic --------------------------------------------------------------

cplx c_sinh(cplx z){
    return c_make( sinh(z.re)*cos(z.im),
                   cosh(z.re)*sin(z.im) );
}

cplx c_cosh(cplx z){
    return c_make( cosh(z.re)*cos(z.im),
                   sinh(z.re)*sin(z.im) );
}

cplx c_tanh(cplx z){
    return c_div(c_sinh(z), c_cosh(z));
}

// --- Sphere Slope (C² line slope) --------------------------------------------

cplx sphere_slope(float theta, float phi){
    float r = tan(phi * 0.5);
    return c_make(r*cos(theta), r*sin(theta));
}

// --- Divide complex by real --------------------------------------------------

cplx c_div_real(cplx z, float m){
    return c_make(z.re/m, z.im/m);
}

// Produces unit-phase complex a/|a|
cplx c_phase_unit(cplx a, float m){
    if (m == 0.0) return c_make(1.0, 0.0);
    return c_make(a.re/m, a.im/m);
}

// --- Complex Line in C² ------------------------------------------------------
cplx2 c2_line(float theta, float phi, float t){
    cplx a = sphere_slope(theta, phi);
    cplx z1 = c_make(cos(t), sin(t));
    cplx z2 = c_mul(a, z1);
    return c2_make(z1, z2);
}

// --- Hopf circle on S³ --------------------------------------------------------

cplx2 hopf_circle(float theta, float phi, float t){
    cplx a = sphere_slope(theta, phi);
    float m = c_mag(a);

    float s = m / sqrt(1.0 + m*m);
    float c = 1.0 / sqrt(1.0 + m*m);

    cplx phase_a = c_phase_unit(a, m);
    cplx e_it = c_make(cos(t), sin(t));

    cplx z1 = c_scale(e_it, c);
    cplx z2 = c_mul(c_scale(phase_a, s), e_it);

    return c2_make(z1, z2);
}

//----Sphere pos to Orientation--------------------------

vec3 sphere_tangent_u(vec3 sp){
    return vec3_normalize(sp);
}

vec3 sphere_tangent_e1(vec3 sp, vec3 north){
    vec3 u = sphere_tangent_u(sp);
    vec3 a = north - dot(north,u);
    vec3 b = a * u;
    return vec3_normalize(b);
}

vec3 sphere_tangent_e2(vec3 e1, vec3 u){
    return vec3_normalize(cross(u,e1));
}

//------- TORUS -----------------------------------------------

# torus_point: given c = |z1|, s = |z2|, and angles u,v return stereographic image in R3
vec3 torus_point(float c, float s, float u, float v){
    float eps = 1 / (1 << 16);
    float denom = 1.0 - s * sin(v);
    # protect against denom ~ 0 numerically
    if(abs(denom) < eps){
        d = (denom >= 0.0) ? eps : -eps;
        denom = d;
    }
    float x = (c * cos(u)) / denom
    float y = (c * sin(u)) / denom
    float z = (s * cos(v)) / denom
    return vec3(x,y,z);
}

vec3 torus_villarceau_left(float c, float s, float alpha, float t){
    return torus_point_vec(c, s, t, t + alpha);
}

vec3 torus_villarceau_right(float c, float s, float alpha, float t){
    return torus_point_vec(c, s, t, -t + alpha);
}

vec3 torus_meridian(float c, float s, float u, float t){
    return torus_point(c, s, u, t);
}

vec3 torus_paralell(float c, float s, float v, float t){
    return torus_point(c, s, t, v);
}

//--- Projection & Perspective -------------------------------------------------

vec3 world_to_perspective_point(vec3 wp, persp p){
    vec3 q = wp - p.pos;
    float x = dot(q, p.right);
    float y = dot(q, p.up);
    float z = dot(q, p.frwd);
    return vec3(x,y,z);
}

vec2 persp_project_pt(vec3 wp, persp p, float focal){
    vec3 pt = world_to_perspective_point(wp, p);
    return vec2(focal * pt.x / pt.z, focal * pt.y / pt.z);
}

// --- Hyperplane Projection ----------------------------------------------------

vec3 projectPointToHyperplane(vec3 view_p,
                              vec3 target_p,
                              vec3 normal,
                              float d)
{
    vec3 look_dir = view_p - target_p;  // FIXED: you used p-v earlier
    float denom = dot(normal, look_dir);

    if (abs(denom) < EPS16)
        return view_p;  // degenerate

    float t = (d - dot(normal, view_p)) / denom;
    return view_p + t * look_dir;
}

//----------------------Curvelinear rendering----------------------------------------------------------------

//Return (theta, phi) where theta = angle from forward axis (z), phi = atan2(y,x).
vec2 direction_to_angles(vec3 d){
    vec3 d_norm = vec3_normalize(d);
    float theta = acos(max(-1.0, min(1.0, d_norm.z)));
    float phi = atan(d_norm.y, d_norm.x);
    return vec2(theta,phi);

vec2 fisheye_equidistant(vec3 p,float f){
    vec2 angs = direction_to_angles(p);
    float theta = angs.x;
    float phi = angs.y;
    float r = f * theta;
    return vec2(r*cos(phi),r*sin(phi);
}

vec2 fisheye_equisolid(vec3 p,float f){
    vec2 angs = direction_to_angles(p);
    float theta = angs.x;
    float phi = angs.y;
    float r = 2.0 * f * sin(theta / 2.0);
    return vec2(r * cos(phi), r * sin(phi));
}

vec2 fisheye_stereographic(vec3 p,float f){
    vec2 angs = direction_to_angles(p);
    float theta = angs.x;
    float phi = angs.y;
    float r = 2.0 * f * tan(theta / 2.0);
    return vec2(r*cos(phi),r*sin(phi));
}

vec2 orthographic_onto_disc(vec3 p,float f){
    vec2 angs = direction_to_angles(p);
    float theta = angs.x;
    float phi = angs.y;
    r = f * sin(theta);
    return vec2(r * cos(phi), r * sin(phi));
}

vec2 equirectangular(vec3 p){
    vec3 d_norm = vec3_normalize(p);
    float theta = acos(max(-1.0, min(1.0, d_norm.z)));
    float phi = atan(d_norm.y, d_norm.x);
    float u = (phi + pi) / (2.0 * pi);
    float v = 1.0 - (theta / pi);
    return vec2(u, v);
}

//-----------------------Stereographic Projection--------------------------------
vec3 plane_to_sphere_pos(vec2 pt){
    float x = pt.x;
    float y = pt.y;
    float xx = pow(x,2);
    float yy = pow(y,2);
    float x2 = 2*x;
    float y2 = 2*y;
    float r = 1.0;
    float d = r + xx + yy;
    float pz = -r + xx + yy;
    float rx = x2 / d;
    float ry = y2 / d;
    float rz = pz / d;
    return vec3(rx,ry,rz);
}

sphere_pt plane_to_sphere_pt(vec2 pt){
    vec3 sp = plane_to_sphere_pos(pt);
    float r = length(sp);
    float theta = atan(sp.y, sp.x);
    float phi = acos(sp.z / r);
    sphere_pt = ret;
    ret.x = sp.x;
    ret.y = sp.y;
    ret.z = sp.z;
    ret.r = r;
    ret.theta = theta;
    ret.phi = phi;
    return ret;
}

//------------------------------Program Begin------------------------------

