function V = vec2orth(v)
% VEC2ORTH Create an orthonormal matrix of size q x q based on the q-vector
% v. The first column of V will be v/norm(v).

[Q,R] = qr(v);
V = Q*sign(R(1));
