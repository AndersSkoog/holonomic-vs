in float t;
uniform float coef_a[16];
uniform float coef_b[16];
uniform float coef_phi[16];
uniform float coef_psi[16];
uniform int M;
uniform int L;
uniform int H_CNT;
uniform int res;

out vec3 v_position;

cplx fourier_curve(){
    float r_t = 0.0;
    float theta_t = 0.0;
    for(int k = 0; k < H_CNT; k++){
        float a_k = coef_a[k];
        float b_k = coef_b[k];
        float phi_k = coef_phi[k];
        float psi_k = coef_psi[k];
        float r_ang = (TAU * float(k) * t) + phi_k;
        float theta_ang = (TAU * float(k) * t) + psi_k;
        r_t += a_k * cos(r_ang);
        theta_t += b_k * sin(theta_ang);
    }
    cplx e_it = c_make(cos(theta_t),sin(theta_t));
    //cmplx gamma_t = c_mul(c_make(r_t, 0.0),e_it);
    cplx gamma_t = c_scale(e_it,r_t);
    int vi = gl_VertexID;
    int Li = int(vi/res);
    float rot_ang = TAU * float(Li) / float(L);
    cplx rot = c_make(cos(rot_ang), sin(rot_ang));
    cplx gamma_rot = c_mul(rot, gamma_t);
    return gamma_rot;

    //if(Li > 0){
    //  float rot_ang = TAU * float(Li) / float(L);
    //  cplx rot = c_make(cos(rot_ang), sin(rot_ang));
    //  cplx gamma_rot = c_mul(rot, gamma_t);
    //  return gamma_rot;
    //}
    //else {
    //  return gamma_t;
    //}
}

vec2 fourier_curve_pt(){
    cplx z = fourier_curve();
    return vec2(z.re,z.im);
}


void main(){
    float t_ang = TAU * t;
    vec2 curve_pt = fourier_curve_pt();
    sphere_pt sp = plane_to_sphere_pt(curve_pt);
    cplx2 hcirc = hopf_circle(sp.theta,sp.phi,t_ang);
    float c = c_mag(hcirc.z1);
    float s = c_mag(hcirc.z2);
    //
    vec3 torus_par = torus_paralell(c,s,v?,t_ang); // what should v be?
    vec3 torus_mer = torus_meridian(c,s,u?,t_ang); // what should u be?
    vec3 torus_vill_l = torus_villaceau_left(c,s,aplpha?,t_ang); // what should alpha be?
    vec3 torus_vill_r = torus_villaceau_right(c,s,alpha?,t_ang);  // what should alpha be?
}