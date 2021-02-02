#version 330 core
layout (location = 0) in vec3 aPos;

uniform float time;
varying float[12] sines;
varying float[12] coses;
varying float random;

void main() {
    for(int i = 0; i < sines.length(); i++) {
        sines[i] = sin(time / (i + 1));
        coses[i] = cos(time / (i + 1)); 
    }
    random = fract(sin(dot(vec2(time, -time / 20.0) ,vec2(12.9898,78.233))) * 43758.5453);
    gl_Position = vec4(aPos.xyz, 1.0);
}