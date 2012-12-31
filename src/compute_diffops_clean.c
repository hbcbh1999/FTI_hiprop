#include "util.h"

static void b_abs(const real_T x[3], real_T y[3]);
static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static void b_eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3]);
static int32_T b_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T
  *bs, const emxArray_real_T *ws);
static void b_fix(real_T *x);
static int32_T b_min(const emxArray_int32_T *varargin_1);
static int32_T b_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags);
static void b_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2]);
static void b_polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2], real_T maxprdir[3]);
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
                      cend, const emxArray_real_T *ws);
static void c_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void c_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static void c_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws);
static void c_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2], real_T
  maxprdir[3]);
static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols);

static void eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3]);
static void eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws);
static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static void gen_vander_univar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static void linfit_lhf_grad_surf_point(const int32_T ngbvs[128], const
  emxArray_real_T *us, const real_T t1[3], const real_T t2[3], const
  emxArray_real_T *nrms, const emxArray_real_T *ws, real_T hess[3]);
static real_T norm2_vec(const emxArray_real_T *v);
static int32_T c_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags, const int32_T ngbfs[256]);
static void polyfit_lhf_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, const
  emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T
  ring, boolean_T iterfit, emxArray_real_T *nrms, emxArray_real_T *curs,
  emxArray_real_T *prdirs);
static void polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128], int32_T
  nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor, int32_T
  degree, real_T nrm[3], int32_T *deg);
static void polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2]);
static int32_T qr_safeguarded(emxArray_real_T *A, int32_T ncols, emxArray_real_T
  *D);
static void rescale_matrix(emxArray_real_T *V, int32_T ncols, emxArray_real_T
  *ts);
static real_T sum(const emxArray_real_T *x);

/* Function Definitions */

/*
 *
 */
static void b_abs(const real_T x[3], real_T y[3])
{
  int32_T k;
  for (k = 0; k < 3; k++) {
    y[k] = fabs(x[k]);
  }
}

static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions)
{
  emxArray_int32_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
  emxArray = *pEmxArray;
  emxArray->data = (int32_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions)
{
  emxArray_real_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
  emxArray = *pEmxArray;
  emxArray->data = (real_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

/*
 * function [curvs, dir, Jinv] = eval_curvature_lhf_surf( grad, H)
 */
static void b_eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3])
{
  real_T grad_sqnorm;
  real_T y;
  real_T tmp;
  real_T grad_norm;
  real_T ell;
  real_T c;
  real_T s;
  real_T v[2];
  real_T d1[2];
  int32_T ix;
  int32_T iy;
  int32_T k;
  real_T a[2];
  real_T W[4];
  real_T U[6];
  int32_T b_ix;
  int32_T b_iy;
  int32_T c_ix;
  int32_T c_iy;

  /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
  /* and pseudo-inverse. */
  /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
  /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
  /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
  /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
  /*  height function in 2x2 H with a local coordinate frame. */
  /*  */
  /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
  /* 'eval_curvature_lhf_surf:12' grad_sqnorm = grad(1)^2+grad(2)^2; */
  grad_sqnorm = grad[0];
  y = pow(grad_sqnorm, 2.0);
  grad_sqnorm = grad[1];
  tmp = pow(grad_sqnorm, 2.0);
  grad_sqnorm = y + tmp;

  /* 'eval_curvature_lhf_surf:13' grad_norm = sqrt(grad_sqnorm); */
  grad_norm = sqrt(grad_sqnorm);

  /*  Compute key parameters */
  /* 'eval_curvature_lhf_surf:16' ell = sqrt(1+grad_sqnorm); */
  ell = sqrt(1.0 + grad_sqnorm);

  /* 'eval_curvature_lhf_surf:17' ell2=1+grad_sqnorm; */
  /* 'eval_curvature_lhf_surf:17' ell3 = ell*(1+grad_sqnorm); */
  /* 'eval_curvature_lhf_surf:18' if grad_norm==0 */
  if (grad_norm == 0.0) {
    /* 'eval_curvature_lhf_surf:19' c = 1; */
    c = 1.0;

    /* 'eval_curvature_lhf_surf:19' s=0; */
    s = 0.0;
  } else {
    /* 'eval_curvature_lhf_surf:20' else */
    /* 'eval_curvature_lhf_surf:21' c = grad(1)/grad_norm; */
    c = grad[0] / grad_norm;

    /* 'eval_curvature_lhf_surf:21' s = grad(2)/grad_norm; */
    s = grad[1] / grad_norm;
  }

  /*  Compute mean curvature and Gaussian curvature */
  /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
  /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
  /*  Solve quadratic equation to compute principal curvatures */
  /* 'eval_curvature_lhf_surf:29' v = [c*H(1,1)+s*H(1,2) c*H(1,2)+s*H(2,2)]; */
  v[0] = c * H[0] + s * H[2];
  v[1] = c * H[2] + s * H[3];

  /* 'eval_curvature_lhf_surf:30' W1 = [v*[c; s]/ell3, v*[-s; c]/ell2]; */
  d1[0] = c;
  d1[1] = s;
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    y += v[ix] * d1[iy];
    ix++;
    iy++;
  }

  d1[0] = -s;
  d1[1] = c;
  tmp = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    tmp += v[ix] * d1[iy];
    ix++;
    iy++;
  }

  v[0] = y / (ell * (1.0 + grad_sqnorm));
  v[1] = tmp / (1.0 + grad_sqnorm);

  /* 'eval_curvature_lhf_surf:31' W = [W1; W1(2) [c*H(1,2)-s*H(1,1), c*H(2,2)-s*H(1,2)]*[-s; c]/ell]; */
  a[0] = c * H[2] - s * H[0];
  a[1] = c * H[3] - s * H[2];
  d1[0] = -s;
  d1[1] = c;
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    y += a[ix] * d1[iy];
    ix++;
    iy++;
    W[k << 1] = v[k];
  }

  W[1] = v[1];
  W[3] = y / ell;

  /*  Lambda = eig(W); */
  /* 'eval_curvature_lhf_surf:34' kH2 = W(1,1)+W(2,2); */
  grad_sqnorm = W[0] + W[3];

  /* 'eval_curvature_lhf_surf:35' tmp = sqrt((W(1,1)-W(2,2))*(W(1,1)-W(2,2))+4*W(1,2)*W(1,2)); */
  tmp = sqrt((W[0] - W[3]) * (W[0] - W[3]) + 4.0 * W[2] * W[2]);

  /* 'eval_curvature_lhf_surf:36' if kH2>0 */
  if (grad_sqnorm > 0.0) {
    /* 'eval_curvature_lhf_surf:37' curvs = 0.5*[kH2+tmp; kH2-tmp]; */
    curvs[0] = 0.5 * (grad_sqnorm + tmp);
    curvs[1] = 0.5 * (grad_sqnorm - tmp);
  } else {
    /* 'eval_curvature_lhf_surf:38' else */
    /* 'eval_curvature_lhf_surf:39' curvs = 0.5*[kH2-tmp; kH2+tmp]; */
    curvs[0] = 0.5 * (grad_sqnorm - tmp);
    curvs[1] = 0.5 * (grad_sqnorm + tmp);
  }

  /* 'eval_curvature_lhf_surf:42' if nargout > 1 */
  /*  Compute principal directions, first with basis of left  */
  /*  singular vectors of Jacobian */
  /*  Compute principal directions in 3-D space */
  /* 'eval_curvature_lhf_surf:47' U = [c/ell -s; s/ell c; grad_norm/ell 0]; */
  U[0] = c / ell;
  U[3] = -s;
  U[1] = s / ell;
  U[4] = c;
  U[2] = grad_norm / ell;
  U[5] = 0.0;

  /* 'eval_curvature_lhf_surf:49' if curvs(1)==curvs(2) */
  if (curvs[0] == curvs[1]) {
    /* 'eval_curvature_lhf_surf:50' dir = U(:,1); */
    for (ix = 0; ix < 3; ix++) {
      dir[ix] = U[ix];
    }
  } else {
    /* 'eval_curvature_lhf_surf:51' else */
    /* 'eval_curvature_lhf_surf:52' if abs(W(1,1)-curvs(2))>abs(W(1,1)-curvs(1)) */
    if (fabs(W[0] - curvs[1]) > fabs(W[0] - curvs[0])) {
      /* 'eval_curvature_lhf_surf:53' d1 = [W(1,1)-curvs(2); W(1,2)]; */
      d1[0] = W[0] - curvs[1];
      d1[1] = W[2];
    } else {
      /* 'eval_curvature_lhf_surf:54' else */
      /* 'eval_curvature_lhf_surf:55' d1 = [-W(1,2); W(1,1)-curvs(1)]; */
      d1[0] = -W[2];
      d1[1] = W[0] - curvs[0];
    }

    /* 'eval_curvature_lhf_surf:58' d1 = d1/sqrt(d1'*d1); */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      y += d1[ix] * d1[iy];
      ix++;
      iy++;
    }

    grad_sqnorm = sqrt(y);
    for (ix = 0; ix < 2; ix++) {
      d1[ix] /= grad_sqnorm;
    }

    /* 'eval_curvature_lhf_surf:59' dir = [U(1,:)*d1; U(2,:)*d1; U(3,:)*d1]; */
    y = 0.0;
    ix = 0;
    iy = 0;
    tmp = 0.0;
    b_ix = 0;
    b_iy = 0;
    grad_sqnorm = 0.0;
    c_ix = 0;
    c_iy = 0;
    for (k = 0; k < 2; k++) {
      y += U[3 * ix] * d1[iy];
      ix++;
      iy++;
      tmp += U[1 + 3 * b_ix] * d1[b_iy];
      b_ix++;
      b_iy++;
      grad_sqnorm += U[2 + 3 * c_ix] * d1[c_iy];
      c_ix++;
      c_iy++;
    }

    dir[0] = y;
    dir[1] = tmp;
    dir[2] = grad_sqnorm;
  }

  /* 'eval_curvature_lhf_surf:62' if nargout>2 */
}

/*
 * function [bs, degree] = eval_vander_bivar(us, bs, degree, ws, interp0, guardosc)
 */
static int32_T b_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T
  *bs, const emxArray_real_T *ws)
{
  int32_T degree;
  emxArray_real_T *V;
  int32_T npnts;
  int32_T ii;
  int32_T jj;
  emxArray_real_T *b_V;
  int32_T c_V;
  int32_T i4;
  int32_T loop_ub;
  int32_T b_loop_ub;
  emxArray_real_T *ws1;
  emxArray_real_T *D;
  emxInit_real_T(&V, 2);

  /* EVAL_VANDER_BIVAR Evaluate generalized Vandermonde matrix. */
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC) */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*  */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /* 'eval_vander_bivar:10' coder.extrinsic('save') */
  /* 'eval_vander_bivar:11' degree = int32(degree); */
  degree = 1;

  /* 'eval_vander_bivar:12' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:15' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:16' if nargin<5 */
  /* 'eval_vander_bivar:17' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:20' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  /* 'eval_vander_bivar:21' while npnts<ncols && degree>1 */
  /* % Construct matrix */
  /* 'eval_vander_bivar:27' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, 1, V);

  /* 'eval_vander_bivar:28' if interp0 */
  /* 'eval_vander_bivar:28' V=V(:,2:end); */
  ii = V->size[1];
  if (2 > ii) {
    jj = 0;
    ii = 0;
  } else {
    jj = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i4 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = ii - jj;
  emxEnsureCapacity((emxArray__common *)b_V, i4, (int32_T)sizeof(real_T));
  loop_ub = (ii - jj) - 1;
  for (ii = 0; ii <= loop_ub; ii++) {
    b_loop_ub = c_V - 1;
    for (i4 = 0; i4 <= b_loop_ub; i4++) {
      b_V->data[i4 + b_V->size[0] * ii] = V->data[i4 + V->size[0] * (jj + ii)];
    }
  }

  ii = V->size[0] * V->size[1];
  V->size[0] = b_V->size[0];
  V->size[1] = b_V->size[1];
  emxEnsureCapacity((emxArray__common *)V, ii, (int32_T)sizeof(real_T));
  loop_ub = b_V->size[1] - 1;
  for (ii = 0; ii <= loop_ub; ii++) {
    b_loop_ub = b_V->size[0] - 1;
    for (jj = 0; jj <= b_loop_ub; jj++) {
      V->data[jj + V->size[0] * ii] = b_V->data[jj + b_V->size[0] * ii];
    }
  }

  emxFree_real_T(&b_V);

  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar:31' if nargin>3 && ~isempty(ws) */
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:32' if degree>2 */
    /* 'eval_vander_bivar:56' else */
    /* 'eval_vander_bivar:57' for ii=1:npnts */
    for (ii = 0; ii + 1 <= npnts; ii++) {
      /* 'eval_vander_bivar:58' for jj=1:ncols */
      for (jj = 0; jj + 1 < 3; jj++) {
        /* 'eval_vander_bivar:58' V(ii,jj) = V(ii,jj) * ws(ii); */
        V->data[ii + V->size[0] * jj] *= ws->data[ii];
      }

      /* 'eval_vander_bivar:59' for jj=1:int32(size(bs,2)) */
      for (jj = 0; jj < 2; jj++) {
        /* 'eval_vander_bivar:59' bs(ii,jj) = bs(ii,jj) * ws(ii); */
        bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
      }
    }
  }

  b_emxInit_real_T(&ws1, 1);
  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:66' ts = coder.nullcopy(zeros(ncols,1)); */
  ii = ws1->size[0];
  ws1->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:67' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, 2, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:70' D = coder.nullcopy(zeros(ncols,1)); */
  ii = D->size[0];
  D->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)D, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:71' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  ii = qr_safeguarded(V, 2, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:74' ncols_sub = ncols; */
  /* 'eval_vander_bivar:75' while rnk < ncols_sub */
  if (ii < 2) {
    /* 'eval_vander_bivar:76' degree = degree-1; */
    degree = 0;

    /* 'eval_vander_bivar:78' if degree==0 */
    /*  Matrix is singular. Consider surface as flat. */
    /* 'eval_vander_bivar:80' bs(:) = 0; */
    ii = bs->size[0] * bs->size[1];
    bs->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)bs, ii, (int32_T)sizeof(real_T));
    for (ii = 0; ii < 2; ii++) {
      loop_ub = bs->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        bs->data[jj + bs->size[0] * ii] = 0.0;
      }
    }
  } else {
    /* % Compute Q'bs */
    /* 'eval_vander_bivar:86' bs = compute_qtb( V, bs, ncols_sub); */
    compute_qtb(V, bs, 2);

    /* % Perform backward substitution and scale the solutions. */
    /* 'eval_vander_bivar:89' for i=1:ncols_sub */
    for (ii = 0; ii + 1 < 3; ii++) {
      /* 'eval_vander_bivar:89' V(i,i) = D(i); */
      V->data[ii + V->size[0] * ii] = D->data[ii];
    }

    /* 'eval_vander_bivar:90' if guardosc */
    /* 'eval_vander_bivar:92' else */
    /* 'eval_vander_bivar:93' bs = backsolve(V, bs, ncols_sub, ts); */
    backsolve(V, bs, 2, ws1);
  }

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
  return degree;
}

/*
 *
 */
static void b_fix(real_T *x)
{
  if (*x > 0.0) {
    *x = floor(*x);
  } else {
    *x = ceil(*x);
  }
}

/*
 *
 */
static int32_T b_min(const emxArray_int32_T *varargin_1)
{
  int32_T minval;
  int32_T n;
  int32_T ix;
  n = varargin_1->size[0];
  minval = varargin_1->data[0];
  if (n == 1) {
  } else {
    for (ix = 1; ix + 1 <= n; ix++) {
      if (varargin_1->data[ix] < minval) {
        minval = varargin_1->data[ix];
      }
    }
  }

  return minval;
}

/*
 * function [ngbvs, nverts, vtags, ftags, ngbfs, nfaces] = obtain_nring_surf...
 *     ( vid, ring, minpnts, tris, opphes, v2he, ngbvs, vtags, ftags, ngbfs)
 */
static int32_T b_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags)
{
  int32_T nverts;
  int32_T fid;
  int32_T lid;
  int32_T nfaces;
  boolean_T overflow;
  boolean_T b4;
  int32_T fid_in;
  static const int8_T iv18[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv19[3] = { 3, 1, 2 };

  int32_T ngbfs[256];
  int32_T opp;
  int32_T nverts_pre;
  int32_T nfaces_pre;
  real_T ring_full;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T b5;
  boolean_T isfirst;
  int32_T exitg3;
  boolean_T guard2 = FALSE;

  /* OBTAIN_NRING_SURF Collect n-ring vertices and faces of a triangle mesh. */
  /*  [NGBVS,NVERTS,VTAGS,FTAGS,NGBFS,NFACES] = OBTAIN_NRING_SURF(VID,RING, ... */
  /*  MINPNTS,TRIS,OPPHES,V2HE,NGBVS,VTAGS,FTAGS,NGBFS)  Collects n-ring */
  /*  vertices and faces of a vertex and saves them into NGBVS and NGBFS, */
  /*  where n is a floating point number with 0.5 increments (1, 1.5, 2, etc.) */
  /*  We define the n-ring verticse as follows: */
  /*   - 0-ring: vertex itself */
  /*   - k-ring vertices: vertices that share an edge with (k-1)-ring vertices */
  /*   - (k+0.5)-ring vertices: k-ring plus vertices that share an element */
  /*            with two vertices of k-ring vertices. */
  /*  For triangle meshes, the k-ring vertices always form some triangles. */
  /*  */
  /*  Input arguments */
  /*    vid: vertex ID */
  /*    ring: the desired number of rings (it is a float as it can have halves) */
  /*    minpnts: the minimum number of points desired */
  /*    tris: element connectivity */
  /*    opphes: opposite half-edges */
  /*    v2he: vertex-to-halfedge mapping */
  /*    ngbvs: buffer space for neighboring vertices (not including vid itself) */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*  */
  /*  Output arguments */
  /*    ngbvs: buffer space for neighboring vertices */
  /*    nverts: number of vertices in the neighborhood */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*    nfaces: number of elements in the neighborhood */
  /*  */
  /*  Notes */
  /*   1. vtags and ftags must be set to false at input. They are reset to */
  /*      false at output. */
  /*   2. Since the vertex itself is always in ring, we do not include it in */
  /*      the output array ngbvs. */
  /*   3. If NGBVS or NGBFS is not enough to store the whole neighborhood, */
  /*      then only a subset of the neighborhood will be returned. */
  /*      The maximum number of points returned is numel(NGBVS) if NGBVS is */
  /*      given as part of the input, or 128 if not an input arguement. */
  /*      The maximum number of faces returned is numel(NGBFS) if NGBFS is */
  /*      given as part of the input, or 256 if not an input arguement. */
  /*  */
  /*  See also OBTAIN_NRING_SURF, OBTAIN_NRING_QUAD, OBTAIN_NRING_CURV, OBTAIN_NRING_VOL */
  /* 'obtain_nring_surf:49' coder.extrinsic('warning'); */
  /*  assert(isscalar(vid)&&isa(vid,'int32')); */
  /*  assert(isscalar(ring)&&isa(ring,'double')); */
  /*  assert(isscalar(minpnts)&&isa(minpnts,'int32')); */
  /*  assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /*  assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /*  assert((size(ngbvs,2)==1) && (size(ngbvs,1)>=1) && isa(ngbvs,'int32')); */
  /*  assert((size(vtags,2)==1) && (size(vtags,1)>=1) && isa(vtags,'logical')); */
  /*  assert((size(ftags,2)==1) && (size(ftags,1)>=1) && isa(ftags,'logical')); */
  /*  assert((size(ngbfs,2)==1) && (size(ngbfs,1)>=1) && isa(ngbfs,'int32')); */
  /* 'obtain_nring_surf:61' MAXNPNTS = int32(128); */
  /* 'obtain_nring_surf:63' assert(ring>=1 && floor(ring*2)==ring*2); */
  /* 'obtain_nring_surf:64' if nargin>=8 */
  /* 'obtain_nring_surf:64' assert( islogical( vtags)); */
  /* 'obtain_nring_surf:65' if nargin>=9 */
  /* 'obtain_nring_surf:65' assert( islogical(ftags)); */
  /* 'obtain_nring_surf:67' fid = heid2fid(v2he(vid)); */
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  /* 'heid2fid:3' coder.inline('always'); */
  /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
  fid = (int32_T)((uint32_T)v2he->data[vid - 1] >> 2U) - 1;

  /* 'obtain_nring_surf:67' lid = heid2leid(v2he(vid)); */
  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  /* 'heid2leid:3' coder.inline('always'); */
  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
  lid = (int32_T)((uint32_T)v2he->data[vid - 1] & 3U);

  /* 'obtain_nring_surf:68' nverts=int32(0); */
  nverts = 0;

  /* 'obtain_nring_surf:68' nfaces=int32(0); */
  nfaces = 0;

  /* 'obtain_nring_surf:68' overflow = false; */
  overflow = FALSE;

  /* 'obtain_nring_surf:70' if ~fid */
  if (!(fid + 1 != 0)) {
  } else {
    /* 'obtain_nring_surf:72' prv = int32([3 1 2]); */
    /* 'obtain_nring_surf:73' nxt = int32([2 3 1]); */
    /* 'obtain_nring_surf:75' if nargin>=7 && ~isempty(ngbvs) */
    /* 'obtain_nring_surf:76' maxnv = int32(numel(ngbvs)); */
    /* 'obtain_nring_surf:81' if nargin>=10 && ~isempty(ngbfs) */
    /* 'obtain_nring_surf:83' else */
    /* 'obtain_nring_surf:84' maxnf = 2*MAXNPNTS; */
    /* 'obtain_nring_surf:84' ngbfs = coder.nullcopy(zeros(maxnf,1, 'int32')); */
    /* 'obtain_nring_surf:87' oneringonly = ring==1 && minpnts==0 && nargout<5; */
    if ((ring == 1.0) && (minpnts == 0)) {
      b4 = TRUE;
    } else {
      b4 = FALSE;
    }

    /* 'obtain_nring_surf:88' hebuf = coder.nullcopy(zeros(maxnv,1, 'int32')); */
    /*  Optimized version for collecting one-ring vertices */
    /* 'obtain_nring_surf:91' if opphes( fid, lid) */
    if (opphes->data[fid + opphes->size[0] * lid] != 0) {
      /* 'obtain_nring_surf:92' fid_in = fid; */
      fid_in = fid + 1;
    } else {
      /* 'obtain_nring_surf:93' else */
      /* 'obtain_nring_surf:94' fid_in = int32(0); */
      fid_in = 0;

      /* 'obtain_nring_surf:96' v = tris(fid, nxt(lid)); */
      /* 'obtain_nring_surf:97' nverts = int32(1); */
      nverts = 1;

      /* 'obtain_nring_surf:97' ngbvs( 1) = v; */
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv18[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      if (!b4) {
        /* 'obtain_nring_surf:99' hebuf(1) = 0; */
        hebuf[0] = 0;
      }
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    /* 'obtain_nring_surf:103' while 1 */
    do {
      exitg4 = 0U;

      /*  Insert vertx into list */
      /* 'obtain_nring_surf:105' lid_prv = prv(lid); */
      lid = iv19[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        if (!b4) {
          /*  Save starting position for next vertex */
          /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv19[lid] -
            1)];

          /* 'obtain_nring_surf:114' nfaces = nfaces + 1; */
          nfaces++;

          /* 'obtain_nring_surf:114' ngbfs( nfaces) = fid; */
          ngbfs[nfaces - 1] = fid + 1;
        }
      } else {
        /* 'obtain_nring_surf:116' else */
        /* 'obtain_nring_surf:117' overflow = true; */
        overflow = TRUE;
      }

      /* 'obtain_nring_surf:120' opp = opphes(fid, lid_prv); */
      opp = opphes->data[fid + opphes->size[0] * lid];

      /* 'obtain_nring_surf:121' fid = heid2fid(opp); */
      /*  HEID2FID   Obtains face ID from half-edge ID. */
      /* 'heid2fid:3' coder.inline('always'); */
      /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
      fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] * lid] >> 2U)
        - 1;

      /* 'obtain_nring_surf:123' if fid == fid_in */
      if (fid + 1 == fid_in) {
        exitg4 = 1U;
      } else {
        /* 'obtain_nring_surf:125' else */
        /* 'obtain_nring_surf:126' lid = heid2leid(opp); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        lid = (int32_T)((uint32_T)opp & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    /* 'obtain_nring_surf:130' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
    if ((ring == 1.0) && ((nverts >= minpnts) || (nverts >= 128) || (nfaces >=
          256))) {
      /* 'obtain_nring_surf:131' if overflow */
    } else {
      /* 'obtain_nring_surf:137' vtags(vid) = true; */
      vtags->data[vid - 1] = TRUE;

      /* 'obtain_nring_surf:138' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_surf:138' vtags(ngbvs(i))=true; */
        vtags->data[ngbvs[lid - 1] - 1] = TRUE;
      }

      /* 'obtain_nring_surf:139' for i=1:nfaces */
      for (lid = 1; lid <= nfaces; lid++) {
        /* 'obtain_nring_surf:139' ftags(ngbfs(i))=true; */
        ftags->data[ngbfs[lid - 1] - 1] = TRUE;
      }

      /*  Define buffers and prepare tags for further processing */
      /* 'obtain_nring_surf:142' nverts_pre = int32(0); */
      nverts_pre = 0;

      /* 'obtain_nring_surf:143' nfaces_pre = int32(0); */
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      /* 'obtain_nring_surf:146' ring_full = fix( ring); */
      ring_full = ring;
      b_fix(&ring_full);

      /* 'obtain_nring_surf:147' minpnts = min(minpnts, maxnv); */
      minpnts = minpnts <= 128 ? minpnts : 128;

      /* 'obtain_nring_surf:149' cur_ring=1; */
      cur_ring = 1.0;

      /* 'obtain_nring_surf:150' while true */
      do {
        exitg1 = 0U;

        /* 'obtain_nring_surf:151' if cur_ring>ring_full || (cur_ring==ring_full && ring_full~=ring) */
        guard1 = FALSE;
        if ((cur_ring > ring_full) || ((cur_ring == ring_full) && (ring_full !=
              ring))) {
          /*  Collect halfring */
          /* 'obtain_nring_surf:153' nfaces_last = nfaces; */
          opp = nfaces;

          /* 'obtain_nring_surf:153' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_surf:154' for ii = nfaces_pre+1 : nfaces_last */
          while (nfaces_pre + 1 <= opp) {
            /*  take opposite vertex in opposite face */
            /* 'obtain_nring_surf:156' for jj=int32(1):3 */
            lid = 0;
            exitg2 = 0U;
            while ((exitg2 == 0U) && (lid + 1 < 4)) {
              /* 'obtain_nring_surf:157' oppe = opphes( ngbfs(ii), jj); */
              /* 'obtain_nring_surf:158' fid = heid2fid(oppe); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                opphes->size[0] * lid) - 1] >> 2U) - 1;

              /* 'obtain_nring_surf:160' if oppe && ~ftags(fid) */
              if ((opphes->data[(ngbfs[nfaces_pre] + opphes->size[0] * lid) - 1]
                   != 0) && (!ftags->data[fid])) {
                /* 'obtain_nring_surf:161' lid = heid2leid(oppe); */
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                /* 'heid2leid:3' coder.inline('always'); */
                /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);

                /* 'obtain_nring_surf:162' v = tris( fid, prv(lid)); */
                /* 'obtain_nring_surf:164' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:165'                         ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  (iv19[lid] - 1)] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:166' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:167' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:167' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:168' ftags(fid) = true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:171' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv19[lid] -
                      1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv19[lid]
                    - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  vtags->data[tris->data[fid + tris->size[0] * (iv19[lid] - 1)]
                    - 1] = TRUE;
                }

                exitg2 = 1U;
              } else {
                lid++;
              }
            }

            nfaces_pre++;
          }

          /* 'obtain_nring_surf:180' if nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nfaces==nfaces_last */
          if ((nverts >= minpnts) || (nfaces >= 256) || (nfaces == opp)) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:182' else */
            /*  If needs to expand, then undo the last half ring */
            /* 'obtain_nring_surf:184' for i=nverts_last+1:nverts */
            for (lid = nverts_last; lid + 1 <= nverts; lid++) {
              /* 'obtain_nring_surf:184' vtags(ngbvs(i)) = false; */
              vtags->data[ngbvs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:185' nverts = nverts_last; */
            nverts = nverts_last;

            /* 'obtain_nring_surf:187' for i=nfaces_last+1:nfaces */
            for (lid = opp; lid + 1 <= nfaces; lid++) {
              /* 'obtain_nring_surf:187' ftags(ngbfs(i)) = false; */
              ftags->data[ngbfs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:188' nfaces = nfaces_last; */
            nfaces = opp;
            guard1 = TRUE;
          }
        } else {
          guard1 = TRUE;
        }

        if (guard1 == TRUE) {
          /*  Collect next full level of ring */
          /* 'obtain_nring_surf:193' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_surf:193' nfaces_pre = nfaces; */
          nfaces_pre = nfaces;

          /* 'obtain_nring_surf:194' for ii=nverts_pre+1 : nverts_last */
          while (nverts_pre + 1 <= nverts_last) {
            /* 'obtain_nring_surf:195' v = ngbvs(ii); */
            /* 'obtain_nring_surf:195' fid = heid2fid(v2he(v)); */
            /*  HEID2FID   Obtains face ID from half-edge ID. */
            /* 'heid2fid:3' coder.inline('always'); */
            /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
            fid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] >> 2U) -
              1;

            /* 'obtain_nring_surf:195' lid = heid2leid(v2he(v)); */
            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            /* 'heid2leid:3' coder.inline('always'); */
            /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
            lid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] & 3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            /* 'obtain_nring_surf:199' allow_early_term = hebuf(ii) && opphes(fid,lid); */
            if ((hebuf[nverts_pre] != 0) && (opphes->data[fid + opphes->size[0] *
                 lid] != 0)) {
              b5 = TRUE;
            } else {
              b5 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b5) {
              /* 'obtain_nring_surf:201' fid = heid2fid(hebuf(ii)); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)hebuf[nverts_pre] >> 2U) - 1;

              /* 'obtain_nring_surf:201' lid = heid2leid(hebuf(ii)); */
              /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
              /* 'heid2leid:3' coder.inline('always'); */
              /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
              lid = (int32_T)((uint32_T)hebuf[nverts_pre] & 3U);
            }

            /*  */
            /* 'obtain_nring_surf:205' if opphes( fid, lid) */
            if (opphes->data[fid + opphes->size[0] * lid] != 0) {
              /* 'obtain_nring_surf:206' fid_in = fid; */
              fid_in = fid + 1;
            } else {
              /* 'obtain_nring_surf:207' else */
              /* 'obtain_nring_surf:208' fid_in = cast(0,class(fid)); */
              fid_in = 0;

              /* 'obtain_nring_surf:210' v = tris(fid, nxt(lid)); */
              /* 'obtain_nring_surf:211' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs); */
              if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                (iv18[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                nverts++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv18[lid]
                  - 1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                vtags->data[tris->data[fid + tris->size[0] * (iv18[lid] - 1)] -
                  1] = TRUE;

                /*  Save starting position for next vertex */
                /* 'obtain_nring_surf:215' hebuf(nverts) = 0; */
                hebuf[nverts - 1] = 0;
              }
            }

            /*  Rotate counterclockwise around the vertex. */
            /* 'obtain_nring_surf:220' isfirst=true; */
            isfirst = TRUE;

            /* 'obtain_nring_surf:221' while true */
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              /* 'obtain_nring_surf:223' lid_prv = prv(lid); */
              lid = iv19[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b5 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /* 'obtain_nring_surf:228' else */
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                /* 'obtain_nring_surf:231' v = tris(fid, lid_prv); */
                /* 'obtain_nring_surf:232' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:233'                     ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  lid] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:235' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * lid] - 1]) &&
                    (!overflow)) {
                  /* 'obtain_nring_surf:236' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:236' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

                  /* 'obtain_nring_surf:236' vtags(v)=true; */
                  vtags->data[tris->data[fid + tris->size[0] * lid] - 1] = TRUE;

                  /*  Save starting position for next ring */
                  /* 'obtain_nring_surf:239' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
                  hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv19[lid] - 1)];
                }

                /* 'obtain_nring_surf:242' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:243' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:243' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:243' ftags(fid)=true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:245' isfirst = false; */
                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                /* 'obtain_nring_surf:248' opp = opphes(fid, lid_prv); */
                opp = opphes->data[fid + opphes->size[0] * lid];

                /* 'obtain_nring_surf:249' fid = heid2fid(opp); */
                /*  HEID2FID   Obtains face ID from half-edge ID. */
                /* 'heid2fid:3' coder.inline('always'); */
                /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid] >> 2U) - 1;

                /* 'obtain_nring_surf:251' if fid == fid_in */
                if (fid + 1 == fid_in) {
                  /*  Finished cycle */
                  exitg3 = 1U;
                } else {
                  /* 'obtain_nring_surf:253' else */
                  /* 'obtain_nring_surf:254' lid = heid2leid(opp); */
                  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                  /* 'heid2leid:3' coder.inline('always'); */
                  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                  lid = (int32_T)((uint32_T)opp & 3U);
                }
              }
            } while (exitg3 == 0U);

            nverts_pre++;
          }

          /* 'obtain_nring_surf:259' cur_ring = cur_ring+1; */
          cur_ring++;

          /* 'obtain_nring_surf:260' if (nverts>=minpnts && cur_ring>=ring) || nfaces==nfaces_pre || overflow */
          if (((nverts >= minpnts) && (cur_ring >= ring)) || (nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:264' nverts_pre = nverts_last; */
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      /* 'obtain_nring_surf:268' vtags(vid) = false; */
      vtags->data[vid - 1] = FALSE;

      /* 'obtain_nring_surf:269' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_surf:269' vtags(ngbvs(i))=false; */
        vtags->data[ngbvs[lid - 1] - 1] = FALSE;
      }

      /* 'obtain_nring_surf:270' if ~oneringonly */
      if (!b4) {
        /* 'obtain_nring_surf:270' for i=1:nfaces */
        for (lid = 1; lid <= nfaces; lid++) {
          /* 'obtain_nring_surf:270' ftags(ngbfs(i))=false; */
          ftags->data[ngbfs[lid - 1] - 1] = FALSE;
        }
      }

      /* 'obtain_nring_surf:271' if overflow */
    }
  }

  return nverts;
}

/*
 * function [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point...
 *     (v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
 */
static void b_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2])
{
  int32_T i;
  int32_T ix;
  real_T absnrm[3];
  static const int8_T iv4[3] = { 0, 1, 0 };

  static const int8_T iv5[3] = { 1, 0, 0 };

  real_T y;
  int32_T b_ix;
  int32_T iy;
  int32_T k;
  real_T grad_norm;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws_row;
  real_T t2[3];
  real_T cs2[3];
  real_T grad[2];
  real_T nrm_l[3];
  real_T P[9];
  real_T b_y;
  int32_T b_iy;
  int32_T c_iy;
  real_T H[4];
  emxArray_real_T *b_us;
  real_T grad_sqnorm;
  real_T ell;
  real_T c;
  real_T s;
  real_T b_v[2];
  real_T a[2];

  /* POLYFIT_LHF_SURF_POINT Compute normal, principal curvatures, and principal */
  /* direction. */
  /*  [NRM,DEG,PRCURVS,MAXPRDIR] = POLYFIT_LHF_SURF_POINT(V,NGBVS,NVERTS,XS, ... */
  /*  NRMS_COOR, DEGREE, INTERP, GUARDOSC) Computes normal NRM, principal */
  /*  curvatures PRCURVS, and principal direction MAXPRDIR at vertex, given list */
  /*  of neighbor points. */
  /*  */
  /*  See also POLYFIT_LHF_SURF */
  /* 'polyfit_lhf_surf_point:12' ngbvs = int32(ngbvs); */
  /* added */
  /* 'polyfit_lhf_surf_point:13' MAXNPNTS=int32(128); */
  /* 'polyfit_lhf_surf_point:14' assert( isa( v, 'int32') && isa( ngbvs, 'int32')); */
  /* 'polyfit_lhf_surf_point:15' assert( isa( nverts, 'int32') && isa( degree, 'int32')); */
  /* 'polyfit_lhf_surf_point:17' if nargin<8 */
  /* 'polyfit_lhf_surf_point:17' guardosc=false; */
  /* 'polyfit_lhf_surf_point:19' if nverts==0 */
  if (nverts == 0) {
    /* 'polyfit_lhf_surf_point:20' nrm = [0; 0; 0]; */
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:20' deg = int32(0); */
    *deg = 0;

    /* 'polyfit_lhf_surf_point:21' prcurvs = [0;0]; */
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:21' maxprdir = [0;0;0]; */
  } else {
    if (nverts >= 128) {
      /* 'polyfit_lhf_surf_point:23' elseif nverts>=MAXNPNTS */
      /* 'polyfit_lhf_surf_point:24' nverts = MAXNPNTS-1; */
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    /* 'polyfit_lhf_surf_point:28' nrm = nrms_coor(v,1:3)'; */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    /* 'polyfit_lhf_surf_point:29' absnrm = abs(nrm); */
    b_abs(nrm, absnrm);

    /* 'polyfit_lhf_surf_point:31' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      /* 'polyfit_lhf_surf_point:32' t1 = [0; 1; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv4[i];
      }
    } else {
      /* 'polyfit_lhf_surf_point:33' else */
      /* 'polyfit_lhf_surf_point:34' t1 = [1; 0; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv5[i];
      }
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 - t1' * nrm * nrm; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * nrm[iy];
      b_ix++;
      iy++;
    }

    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] -= y * nrm[ix];
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 / sqrt(t1'*t1); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * absnrm[iy];
      b_ix++;
      iy++;
    }

    grad_norm = sqrt(y);
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] /= grad_norm;
    }

    emxInit_real_T(&us, 2);
    b_emxInit_real_T(&bs, 1);
    b_emxInit_real_T(&ws_row, 1);

    /* 'polyfit_lhf_surf_point:38' t2 = cross_col( nrm, t1); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    /* 'polyfit_lhf_surf_point:41' us = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:42' bs = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:43' ws_row = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:45' us(1,:)=0; */
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:45' ws_row(1)=1; */
    ws_row->data[0] = 1.0;

    /* 'polyfit_lhf_surf_point:46' for ii=1:nverts */
    for (i = 0; i + 1 <= nverts; i++) {
      /* 'polyfit_lhf_surf_point:47' u = xs(ngbvs(ii),1:3)-xs(v,1:3); */
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      /* 'polyfit_lhf_surf_point:49' us(ii+1-int32(interp),1) = u*t1; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;

      /* 'polyfit_lhf_surf_point:50' us(ii+1-int32(interp),2) = u*t2; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;

      /* 'polyfit_lhf_surf_point:51' bs(ii+1-int32(interp)) = u*nrm; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * nrm[iy];
        b_ix++;
        iy++;
      }

      bs->data[i] = y;

      /*  Compute normal-based weights */
      /* 'polyfit_lhf_surf_point:54' ws_row(ii+1-int32(interp)) = max(0, nrms_coor(ngbvs(ii),:)*nrm); */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += nrms_coor->data[(ngbvs[i] + nrms_coor->size[0] * b_ix) - 1] *
          nrm[iy];
        b_ix++;
        iy++;
      }

      y = 0.0 >= y ? 0.0 : y;
      ws_row->data[i] = y;
    }

    /* 'polyfit_lhf_surf_point:57' if degree==0 */
    if (degree == 0) {
      /*  Use linear fitting without weight */
      /* 'polyfit_lhf_surf_point:59' ws_row(:) = 1; */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      /* 'polyfit_lhf_surf_point:59' degree=int32(1); */
      degree = 1;
    }

    /*  Compute the coefficients */
    /* 'polyfit_lhf_surf_point:63' [bs, deg] = eval_vander_bivar( us, bs, degree, ws_row, interp, guardosc); */
    *deg = degree;
    eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    /* 'polyfit_lhf_surf_point:66' if deg<=1 */
    /* 'polyfit_lhf_surf_point:67' coder.varsize('cs', [6,1],[1,0]); */
    /* 'polyfit_lhf_surf_point:68' cs = bs(2-int32(interp):n); */
    /* 'polyfit_lhf_surf_point:70' grad = [cs(1); cs(2)]; */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];

    /* 'polyfit_lhf_surf_point:71' nrm_l = [-grad; 1]/sqrt(1+grad'*grad); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      y += grad[b_ix] * grad[iy];
      b_ix++;
      iy++;
    }

    grad_norm = sqrt(1.0 + y);
    for (i = 0; i < 2; i++) {
      nrm_l[i] = -grad[i] / grad_norm;
    }

    nrm_l[2] = 1.0 / grad_norm;

    /* 'polyfit_lhf_surf_point:73' P = [t1, t2, nrm]; */
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
    /* 'polyfit_lhf_surf_point:75' nrm = [P(1,:) * nrm_l; P(2,:) * nrm_l; P(3,:) * nrm_l]; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    b_y = 0.0;
    i = 0;
    b_iy = 0;
    grad_norm = 0.0;
    ix = 0;
    c_iy = 0;
    for (k = 0; k < 3; k++) {
      y += P[3 * b_ix] * nrm_l[iy];
      b_ix++;
      iy++;
      b_y += P[1 + 3 * i] * nrm_l[b_iy];
      i++;
      b_iy++;
      grad_norm += P[2 + 3 * ix] * nrm_l[c_iy];
      ix++;
      c_iy++;
    }

    nrm[0] = y;
    nrm[1] = b_y;
    nrm[2] = grad_norm;

    /* 'polyfit_lhf_surf_point:77' if deg>1 */
    if (*deg > 1) {
      /* 'polyfit_lhf_surf_point:78' H = [2*cs(3) cs(4); cs(4) 2*cs(5)]; */
      H[0] = 2.0 * bs->data[2];
      H[2] = bs->data[3];
      H[1] = bs->data[3];
      H[3] = 2.0 * bs->data[4];
    } else if (nverts >= 2) {
      /* 'polyfit_lhf_surf_point:79' elseif deg<=1 && nverts>=2 */
      /* 'polyfit_lhf_surf_point:80' if deg==0 && nverts>=2 */
      if (*deg == 0) {
        emxInit_real_T(&b_us, 2);

        /* 'polyfit_lhf_surf_point:81' us = us(1:3-int32(interp),:); */
        ix = b_us->size[0] * b_us->size[1];
        b_us->size[0] = 2;
        b_us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)b_us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          for (b_iy = 0; b_iy < 2; b_iy++) {
            b_us->data[b_iy + b_us->size[0] * ix] = us->data[b_iy + us->size[0] *
              ix];
          }
        }

        ix = us->size[0] * us->size[1];
        us->size[0] = b_us->size[0];
        us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          i = b_us->size[0] - 1;
          for (b_iy = 0; b_iy <= i; b_iy++) {
            us->data[b_iy + us->size[0] * ix] = b_us->data[b_iy + b_us->size[0] *
              ix];
          }
        }

        emxFree_real_T(&b_us);

        /* 'polyfit_lhf_surf_point:82' ws_row(1:3-int32(interp)) = 1; */
        for (ix = 0; ix < 2; ix++) {
          ws_row->data[ix] = 1.0;
        }
      }

      /*  Try to compute curvatures from normals */
      /* 'polyfit_lhf_surf_point:86' cs2 = linfit_lhf_grad_surf_point( ngbvs, us, t1, t2, nrms_coor, ws_row, interp); */
      linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row, cs2);

      /* 'polyfit_lhf_surf_point:87' H = [cs2(1) cs2(2); cs2(2) cs2(3)]; */
      H[0] = cs2[0];
      H[2] = cs2[1];
      H[1] = cs2[1];
      H[3] = cs2[2];
    } else {
      /* 'polyfit_lhf_surf_point:88' else */
      /* 'polyfit_lhf_surf_point:89' H = coder.nullcopy(zeros(2,2)); */
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);

    /* 'polyfit_lhf_surf_point:92' if deg>=1 */
    if (*deg >= 1) {
      /* 'polyfit_lhf_surf_point:93' if nargout==3 */
      /* 'polyfit_lhf_surf_point:94' prcurvs = eval_curvature_lhf_surf(grad, H); */
      /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
      /* and pseudo-inverse. */
      /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
      /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
      /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
      /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
      /*  height function in 2x2 H with a local coordinate frame. */
      /*  */
      /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
      /* 'eval_curvature_lhf_surf:12' grad_sqnorm = grad(1)^2+grad(2)^2; */
      grad_norm = grad[1];
      y = pow(grad_norm, 2.0);
      grad_norm = grad[0];
      b_y = pow(grad_norm, 2.0);
      grad_sqnorm = b_y + y;

      /* 'eval_curvature_lhf_surf:13' grad_norm = sqrt(grad_sqnorm); */
      grad_norm = sqrt(grad_sqnorm);

      /*  Compute key parameters */
      /* 'eval_curvature_lhf_surf:16' ell = sqrt(1+grad_sqnorm); */
      ell = sqrt(1.0 + grad_sqnorm);

      /* 'eval_curvature_lhf_surf:17' ell2=1+grad_sqnorm; */
      /* 'eval_curvature_lhf_surf:17' ell3 = ell*(1+grad_sqnorm); */
      /* 'eval_curvature_lhf_surf:18' if grad_norm==0 */
      if (grad_norm == 0.0) {
        /* 'eval_curvature_lhf_surf:19' c = 1; */
        c = 1.0;

        /* 'eval_curvature_lhf_surf:19' s=0; */
        s = 0.0;
      } else {
        /* 'eval_curvature_lhf_surf:20' else */
        /* 'eval_curvature_lhf_surf:21' c = grad(1)/grad_norm; */
        c = grad[0] / grad_norm;

        /* 'eval_curvature_lhf_surf:21' s = grad(2)/grad_norm; */
        s = grad[1] / grad_norm;
      }

      /*  Compute mean curvature and Gaussian curvature */
      /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
      /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
      /*  Solve quadratic equation to compute principal curvatures */
      /* 'eval_curvature_lhf_surf:29' v = [c*H(1,1)+s*H(1,2) c*H(1,2)+s*H(2,2)]; */
      b_v[0] = c * H[0] + s * H[2];
      b_v[1] = c * H[2] + s * H[3];

      /* 'eval_curvature_lhf_surf:30' W1 = [v*[c; s]/ell3, v*[-s; c]/ell2]; */
      grad[0] = c;
      grad[1] = s;
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 2; k++) {
        y += b_v[b_ix] * grad[iy];
        b_ix++;
        iy++;
      }

      grad[0] = -s;
      grad[1] = c;
      b_y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 2; k++) {
        b_y += b_v[b_ix] * grad[iy];
        b_ix++;
        iy++;
      }

      b_v[0] = y / (ell * (1.0 + grad_sqnorm));
      b_v[1] = b_y / (1.0 + grad_sqnorm);

      /* 'eval_curvature_lhf_surf:31' W = [W1; W1(2) [c*H(1,2)-s*H(1,1), c*H(2,2)-s*H(1,2)]*[-s; c]/ell]; */
      a[0] = c * H[2] - s * H[0];
      a[1] = c * H[3] - s * H[2];
      grad[0] = -s;
      grad[1] = c;
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 2; k++) {
        y += a[b_ix] * grad[iy];
        b_ix++;
        iy++;
        H[k << 1] = b_v[k];
      }

      H[1] = b_v[1];
      H[3] = y / ell;

      /*  Lambda = eig(W); */
      /* 'eval_curvature_lhf_surf:34' kH2 = W(1,1)+W(2,2); */
      grad_norm = H[0] + H[3];

      /* 'eval_curvature_lhf_surf:35' tmp = sqrt((W(1,1)-W(2,2))*(W(1,1)-W(2,2))+4*W(1,2)*W(1,2)); */
      s = sqrt((H[0] - H[3]) * (H[0] - H[3]) + 4.0 * H[2] * H[2]);

      /* 'eval_curvature_lhf_surf:36' if kH2>0 */
      if (grad_norm > 0.0) {
        /* 'eval_curvature_lhf_surf:37' curvs = 0.5*[kH2+tmp; kH2-tmp]; */
        prcurvs[0] = 0.5 * (grad_norm + s);
        prcurvs[1] = 0.5 * (grad_norm - s);
      } else {
        /* 'eval_curvature_lhf_surf:38' else */
        /* 'eval_curvature_lhf_surf:39' curvs = 0.5*[kH2-tmp; kH2+tmp]; */
        prcurvs[0] = 0.5 * (grad_norm - s);
        prcurvs[1] = 0.5 * (grad_norm + s);
      }

      /* 'eval_curvature_lhf_surf:42' if nargout > 1 */
    } else {
      /* 'polyfit_lhf_surf_point:100' else */
      /* 'polyfit_lhf_surf_point:101' prcurvs = [0;0]; */
      for (i = 0; i < 2; i++) {
        prcurvs[i] = 0.0;
      }

      /* 'polyfit_lhf_surf_point:102' maxprdir = [0;0;0]; */
    }
  }
}

/*
 * function [deg, prcurvs, maxprdir] = polyfit_lhfgrad_surf_point...
 *     ( v, ngbvs, nverts, xs, nrms, degree, interp, guardosc)
 */
static void b_polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2], real_T maxprdir[3])
{
  int32_T i;
  int32_T iy;
  real_T nrm[3];
  int32_T k;
  real_T absnrm[3];
  static const int8_T iv10[3] = { 0, 1, 0 };

  static const int8_T iv11[3] = { 1, 0, 0 };

  real_T y;
  int32_T ix;
  int32_T b_iy;
  real_T h12;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws_row;
  real_T t2[3];
  real_T u[3];
  real_T grad[2];
  real_T H[4];
  real_T P[9];
  real_T b_y;
  int32_T b_ix;
  int32_T c_iy;

  /* POLYFIT_LHFGRAD_SURF_POINT Compute principal curvatures and principal  */
  /* direction. */
  /*  [DEG,PRCURVS,MAXPRDIR] = POLYFIT_LHFGRAD_SURF_POINT(V,NGBVS,NVERTS, ... */
  /*  XS,NRMS,DEGREE,INTERP,GUARDOSC) Computes principal curvatures and */
  /*  principal direction at vertex, using given points XS and vertex normals NRMS. */
  /*  */
  /*  See also POLYFIT_LHFGRAD_SURF_POINT */
  /* 'polyfit_lhfgrad_surf_point:11' MAXNPNTS=int32(128); */
  /* 'polyfit_lhfgrad_surf_point:13' if nargin<8 */
  /* 'polyfit_lhfgrad_surf_point:13' guardosc=false; */
  /* 'polyfit_lhfgrad_surf_point:15' if nverts==0 */
  if (nverts == 0) {
    /* 'polyfit_lhfgrad_surf_point:16' deg = int32(0); */
    *deg = 0;

    /* 'polyfit_lhfgrad_surf_point:16' prcurvs = [0;0]; */
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:16' maxprdir = [0;0;0]; */
    for (i = 0; i < 3; i++) {
      maxprdir[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      /* 'polyfit_lhfgrad_surf_point:18' elseif nverts>=MAXNPNTS */
      /* 'polyfit_lhfgrad_surf_point:19' nverts = MAXNPNTS-1+int32(interp); */
      nverts = 127;
    }

    /*  First, compute the rotation matrix */
    /* 'polyfit_lhfgrad_surf_point:23' nrm = nrms(v,1:3)'; */
    for (iy = 0; iy < 3; iy++) {
      nrm[iy] = nrms->data[(v + nrms->size[0] * iy) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    /* 'polyfit_lhfgrad_surf_point:24' absnrm = abs(nrm); */
    for (k = 0; k < 3; k++) {
      absnrm[k] = fabs(nrm[k]);
    }

    /* 'polyfit_lhfgrad_surf_point:26' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      /* 'polyfit_lhfgrad_surf_point:27' t1 = [0; 1; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv10[i];
      }
    } else {
      /* 'polyfit_lhfgrad_surf_point:28' else */
      /* 'polyfit_lhfgrad_surf_point:29' t1 = [1; 0; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv11[i];
      }
    }

    /* 'polyfit_lhfgrad_surf_point:32' t1 = t1 - t1' * nrm * nrm; */
    y = 0.0;
    ix = 0;
    b_iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[ix] * nrm[b_iy];
      ix++;
      b_iy++;
    }

    for (iy = 0; iy < 3; iy++) {
      absnrm[iy] -= y * nrm[iy];
    }

    /* 'polyfit_lhfgrad_surf_point:32' t1 = t1 / sqrt(t1'*t1); */
    y = 0.0;
    ix = 0;
    b_iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[ix] * absnrm[b_iy];
      ix++;
      b_iy++;
    }

    h12 = sqrt(y);
    for (iy = 0; iy < 3; iy++) {
      absnrm[iy] /= h12;
    }

    emxInit_real_T(&us, 2);
    emxInit_real_T(&bs, 2);
    b_emxInit_real_T(&ws_row, 1);

    /* 'polyfit_lhfgrad_surf_point:33' t2 = cross_col( nrm, t1); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Evaluate local coordinate system and weights */
    /* 'polyfit_lhfgrad_surf_point:36' us = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    iy = us->size[0] * us->size[1];
    us->size[0] = nverts + 1;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, iy, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:37' bs = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    iy = bs->size[0] * bs->size[1];
    bs->size[0] = nverts + 1;
    bs->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)bs, iy, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:38' ws_row = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    iy = ws_row->size[0];
    ws_row->size[0] = nverts + 1;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:40' if ~interp */
    /* 'polyfit_lhfgrad_surf_point:41' us(1,:)=0; */
    for (iy = 0; iy < 2; iy++) {
      us->data[us->size[0] * iy] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:41' bs(1,:)=0; */
    for (iy = 0; iy < 2; iy++) {
      bs->data[bs->size[0] * iy] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:41' ws_row(1) = 1; */
    ws_row->data[0] = 1.0;

    /* 'polyfit_lhfgrad_surf_point:44' for ii=1:nverts */
    for (i = 1; i <= nverts; i++) {
      /* 'polyfit_lhfgrad_surf_point:45' u = xs(ngbvs(ii),1:3)-xs(v,1:3); */
      for (iy = 0; iy < 3; iy++) {
        u[iy] = xs->data[(ngbvs[i - 1] + xs->size[0] * iy) - 1] - xs->data[(v +
          xs->size[0] * iy) - 1];
      }

      /* 'polyfit_lhfgrad_surf_point:47' us(ii+1-int32(interp),1) = u*t1; */
      y = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        y += u[ix] * absnrm[b_iy];
        ix++;
        b_iy++;
      }

      us->data[i] = y;

      /* 'polyfit_lhfgrad_surf_point:48' us(ii+1-int32(interp),2) = u*t2; */
      y = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        y += u[ix] * t2[b_iy];
        ix++;
        b_iy++;
      }

      us->data[i + us->size[0]] = y;

      /* 'polyfit_lhfgrad_surf_point:50' nrm_ii = nrms(ngbvs(ii),1:3); */
      /* 'polyfit_lhfgrad_surf_point:51' w = nrm_ii*nrm; */
      h12 = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        h12 += nrms->data[(ngbvs[i - 1] + nrms->size[0] * ix) - 1] * nrm[b_iy];
        ix++;
        b_iy++;
      }

      /* 'polyfit_lhfgrad_surf_point:53' if w>0 */
      if (h12 > 0.0) {
        /* 'polyfit_lhfgrad_surf_point:54' bs(ii+1-int32(interp),1) = -(nrm_ii*t1)/w; */
        y = 0.0;
        ix = 0;
        b_iy = 0;
        for (k = 0; k < 3; k++) {
          y += nrms->data[(ngbvs[i - 1] + nrms->size[0] * ix) - 1] * absnrm[b_iy];
          ix++;
          b_iy++;
        }

        bs->data[i] = -y / h12;

        /* 'polyfit_lhfgrad_surf_point:55' bs(ii+1-int32(interp),2) = -(nrm_ii*t2)/w; */
        y = 0.0;
        ix = 0;
        b_iy = 0;
        for (k = 0; k < 3; k++) {
          y += nrms->data[(ngbvs[i - 1] + nrms->size[0] * ix) - 1] * t2[b_iy];
          ix++;
          b_iy++;
        }

        bs->data[i + bs->size[0]] = -y / h12;
      }

      /* 'polyfit_lhfgrad_surf_point:57' ws_row(ii+1-int32(interp)) = max(0,w); */
      y = 0.0 >= h12 ? 0.0 : h12;
      ws_row->data[i] = y;
    }

    /* 'polyfit_lhfgrad_surf_point:60' if degree==0 */
    if (degree == 0) {
      /*  Use linear fitting without weight */
      /* 'polyfit_lhfgrad_surf_point:62' ws_row(:) = 1; */
      i = ws_row->size[0];
      iy = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
      i--;
      for (iy = 0; iy <= i; iy++) {
        ws_row->data[iy] = 1.0;
      }

      /* 'polyfit_lhfgrad_surf_point:62' degree=int32(1); */
      degree = 1;
    }

    /*  Compute the coefficients and store them */
    /* 'polyfit_lhfgrad_surf_point:66' [bs, deg] = eval_vander_bivar( us, bs, degree, ws_row, interp, guardosc); */
    *deg = degree;
    c_eval_vander_bivar(us, bs, deg, ws_row);

    /* 'polyfit_lhfgrad_surf_point:68' if interp */
    /* 'polyfit_lhfgrad_surf_point:73' else */
    /*  Convert coefficients into normals and curvatures */
    /* 'polyfit_lhfgrad_surf_point:75' grad = [bs(1,1) bs(1,2)]; */
    grad[0] = bs->data[0];
    grad[1] = bs->data[bs->size[0]];

    /* 'polyfit_lhfgrad_surf_point:76' h12 = 0.5*(bs(3,1)+bs(2,2)); */
    h12 = bs->data[2] + bs->data[1 + bs->size[0]];
    h12 *= 0.5;

    /* 'polyfit_lhfgrad_surf_point:77' H = [bs(2,1) h12; h12 bs(3,2)]; */
    H[0] = bs->data[1];
    H[2] = h12;
    H[1] = h12;
    H[3] = bs->data[2 + bs->size[0]];

    /* 'polyfit_lhfgrad_surf_point:80' if nargout<=2 */
    /* 'polyfit_lhfgrad_surf_point:82' else */
    /* 'polyfit_lhfgrad_surf_point:83' [prcurvs, maxprdir_l] = eval_curvature_lhf_surf(grad, H); */
    b_eval_curvature_lhf_surf(grad, H, prcurvs, u);

    /*  maxprdir = P * maxprdir_l; */
    /* 'polyfit_lhfgrad_surf_point:85' P = [t1, t2, nrm]; */
    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    for (iy = 0; iy < 3; iy++) {
      P[iy] = absnrm[iy];
      P[3 + iy] = t2[iy];
      P[6 + iy] = nrm[iy];
    }

    /* 'polyfit_lhfgrad_surf_point:86' maxprdir = [P(1,:) * maxprdir_l; P(2,:) * maxprdir_l; P(3,:) * maxprdir_l]; */
    y = 0.0;
    ix = 0;
    b_iy = 0;
    h12 = 0.0;
    i = 0;
    iy = 0;
    b_y = 0.0;
    b_ix = 0;
    c_iy = 0;
    for (k = 0; k < 3; k++) {
      y += P[3 * ix] * u[b_iy];
      ix++;
      b_iy++;
      h12 += P[1 + 3 * i] * u[iy];
      i++;
      iy++;
      b_y += P[2 + 3 * b_ix] * u[c_iy];
      b_ix++;
      c_iy++;
    }

    maxprdir[0] = y;
    maxprdir[1] = h12;
    maxprdir[2] = b_y;
  }
}

/*
 * function bs = backsolve(R, bs, cend, ws)
 */
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
                      cend, const emxArray_real_T *ws)
{
  int32_T kk;
  int32_T jj;
  int32_T ii;

  /*  Perform backward substitution. */
  /*      bs = backsolve(R, bs) */
  /*      bs = backsolve(R, bs, cend) */
  /*      bs = backsolve(R, bs, cend, ws) */
  /*   Compute bs = (triu(R(1:cend,1:cend))\bs) ./ ws; */
  /*   The right-hand side vector bs can have multiple columns. */
  /*   By default, cend is size(R,2), and ws is ones. */
  /* 'backsolve:10' if nargin<3 */
  /* 'backsolve:12' for kk=1:int32(size(bs,2)) */
  for (kk = 0; kk < 2; kk++) {
    /* 'backsolve:13' for jj=cend:-1:1 */
    for (jj = cend - 1; jj + 1 > 0; jj--) {
      /* 'backsolve:14' for ii=jj+1:cend */
      for (ii = jj + 1; ii + 1 <= cend; ii++) {
        /* 'backsolve:15' bs(jj,kk) = bs(jj,kk) - R(jj,ii) * bs(ii,kk); */
        bs->data[jj + bs->size[0] * kk] -= R->data[jj + R->size[0] * ii] *
          bs->data[ii + bs->size[0] * kk];
      }

      /* 'backsolve:18' assert( R(jj,jj)~=0); */
      /* 'backsolve:19' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
      bs->data[jj + bs->size[0] * kk] /= R->data[jj + R->size[0] * jj];
    }
  }

  /* 'backsolve:23' if nargin>3 */
  /*  Scale bs back if ts is given. */
  /* 'backsolve:25' for kk=1:int32(size(bs,2)) */
  for (kk = 0; kk < 2; kk++) {
    /* 'backsolve:26' for jj = 1:cend */
    for (jj = 0; jj + 1 <= cend; jj++) {
      /* 'backsolve:27' bs(jj,kk) = bs(jj,kk) / ws(jj); */
      bs->data[jj + bs->size[0] * kk] /= ws->data[jj];
    }
  }
}

static void c_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions)
{
  emxArray_int32_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
  emxArray = *pEmxArray;
  emxArray->data = (int32_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

static void c_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions)
{
  emxArray_real_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
  emxArray = *pEmxArray;
  emxArray->data = (real_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
  }
}

/*
 * function [bs, degree] = eval_vander_bivar(us, bs, degree, ws, interp0, guardosc)
 */
static void c_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws)
{
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  emxArray_real_T *ws1;
  int32_T jj;
  int32_T loop_ub;
  real_T A;
  int32_T ii;
  emxArray_real_T *D;
  int32_T exitg1;

  /* EVAL_VANDER_BIVAR Evaluate generalized Vandermonde matrix. */
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC) */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*  */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /* 'eval_vander_bivar:10' coder.extrinsic('save') */
  /* 'eval_vander_bivar:11' degree = int32(degree); */
  /* 'eval_vander_bivar:12' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:15' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:16' if nargin<5 */
  /* 'eval_vander_bivar:17' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:20' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  ncols = (*degree + 2) * (*degree + 1) / 2;

  /* 'eval_vander_bivar:21' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (*degree > 1)) {
    /* 'eval_vander_bivar:22' degree=degree-1; */
    (*degree)--;

    /* 'eval_vander_bivar:23' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
    ncols = (*degree + 2) * (*degree + 1) / 2;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  /* 'eval_vander_bivar:27' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, *degree, V);

  /* 'eval_vander_bivar:28' if interp0 */
  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar:31' if nargin>3 && ~isempty(ws) */
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:32' if degree>2 */
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      /* 'eval_vander_bivar:33' ws1 = us(:,1).*us(:,1)+us(:,2).*us(:,2); */
      jj = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] = us->data[jj] * us->data[jj] + us->data[jj + us->size[0]]
          * us->data[jj + us->size[0]];
      }

      /* 'eval_vander_bivar:34' ws1 = ws1 + sum(ws1)/double(npnts)*1.e-2; */
      A = sum(ws1);
      A = A / (real_T)npnts * 0.01;
      jj = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] += A;
      }

      /* 'eval_vander_bivar:35' if degree<4 */
      if (*degree < 4) {
        /* 'eval_vander_bivar:36' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:37' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:38' ws1(ii) = ws(ii) / sqrt(ws1(ii)); */
            ws1->data[ii] = ws->data[ii] / sqrt(ws1->data[ii]);
          } else {
            /* 'eval_vander_bivar:39' else */
            /* 'eval_vander_bivar:40' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      } else {
        /* 'eval_vander_bivar:43' else */
        /* 'eval_vander_bivar:44' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:45' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:46' ws1(ii) = ws(ii) / ws1(ii); */
            ws1->data[ii] = ws->data[ii] / ws1->data[ii];
          } else {
            /* 'eval_vander_bivar:47' else */
            /* 'eval_vander_bivar:48' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      }

      /* 'eval_vander_bivar:52' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:53' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:53' V(ii,jj) = V(ii,jj) * ws1(ii); */
          V->data[ii + V->size[0] * jj] *= ws1->data[ii];
        }

        /* 'eval_vander_bivar:54' for jj=1:size(bs,2) */
        for (jj = 0; jj < 2; jj++) {
          /* 'eval_vander_bivar:54' bs(ii,jj) = bs(ii,jj) * ws1(ii); */
          bs->data[ii + bs->size[0] * jj] *= ws1->data[ii];
        }
      }
    } else {
      /* 'eval_vander_bivar:56' else */
      /* 'eval_vander_bivar:57' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:58' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:58' V(ii,jj) = V(ii,jj) * ws(ii); */
          V->data[ii + V->size[0] * jj] *= ws->data[ii];
        }

        /* 'eval_vander_bivar:59' for jj=1:int32(size(bs,2)) */
        for (jj = 0; jj < 2; jj++) {
          /* 'eval_vander_bivar:59' bs(ii,jj) = bs(ii,jj) * ws(ii); */
          bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
        }
      }
    }
  }

  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:66' ts = coder.nullcopy(zeros(ncols,1)); */
  jj = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:67' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:70' D = coder.nullcopy(zeros(ncols,1)); */
  jj = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, jj, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:71' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  ii = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:74' ncols_sub = ncols; */
  /* 'eval_vander_bivar:75' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (ii < ncols) {
      /* 'eval_vander_bivar:76' degree = degree-1; */
      (*degree)--;

      /* 'eval_vander_bivar:78' if degree==0 */
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar:80' bs(:) = 0; */
        jj = bs->size[0] * bs->size[1];
        bs->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)bs, jj, (int32_T)sizeof(real_T));
        for (jj = 0; jj < 2; jj++) {
          loop_ub = bs->size[0] - 1;
          for (ii = 0; ii <= loop_ub; ii++) {
            bs->data[ii + bs->size[0] * jj] = 0.0;
          }
        }

        exitg1 = 1U;
      } else {
        /* 'eval_vander_bivar:82' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U);
      }
    } else {
      /* % Compute Q'bs */
      /* 'eval_vander_bivar:86' bs = compute_qtb( V, bs, ncols_sub); */
      compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar:89' for i=1:ncols_sub */
      for (ii = 0; ii + 1 <= ncols; ii++) {
        /* 'eval_vander_bivar:89' V(i,i) = D(i); */
        V->data[ii + V->size[0] * ii] = D->data[ii];
      }

      /* 'eval_vander_bivar:90' if guardosc */
      /* 'eval_vander_bivar:92' else */
      /* 'eval_vander_bivar:93' bs = backsolve(V, bs, ncols_sub, ts); */
      backsolve(V, bs, ncols, ws1);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
}

/*
 * function [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point...
 *     (v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
 */
static void c_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2], real_T
  maxprdir[3])
{
  int32_T i;
  int32_T ix;
  real_T absnrm[3];
  static const int8_T iv6[3] = { 0, 1, 0 };

  static const int8_T iv7[3] = { 1, 0, 0 };

  real_T y;
  int32_T b_ix;
  int32_T iy;
  int32_T k;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws_row;
  real_T t2[3];
  real_T cs2[3];
  real_T grad[2];
  real_T nrm_l[3];
  real_T P[9];
  real_T b_y;
  int32_T b_iy;
  real_T c_y;
  int32_T c_iy;
  real_T H[4];
  emxArray_real_T *b_us;

  /* POLYFIT_LHF_SURF_POINT Compute normal, principal curvatures, and principal */
  /* direction. */
  /*  [NRM,DEG,PRCURVS,MAXPRDIR] = POLYFIT_LHF_SURF_POINT(V,NGBVS,NVERTS,XS, ... */
  /*  NRMS_COOR, DEGREE, INTERP, GUARDOSC) Computes normal NRM, principal */
  /*  curvatures PRCURVS, and principal direction MAXPRDIR at vertex, given list */
  /*  of neighbor points. */
  /*  */
  /*  See also POLYFIT_LHF_SURF */
  /* 'polyfit_lhf_surf_point:12' ngbvs = int32(ngbvs); */
  /* added */
  /* 'polyfit_lhf_surf_point:13' MAXNPNTS=int32(128); */
  /* 'polyfit_lhf_surf_point:14' assert( isa( v, 'int32') && isa( ngbvs, 'int32')); */
  /* 'polyfit_lhf_surf_point:15' assert( isa( nverts, 'int32') && isa( degree, 'int32')); */
  /* 'polyfit_lhf_surf_point:17' if nargin<8 */
  /* 'polyfit_lhf_surf_point:17' guardosc=false; */
  /* 'polyfit_lhf_surf_point:19' if nverts==0 */
  if (nverts == 0) {
    /* 'polyfit_lhf_surf_point:20' nrm = [0; 0; 0]; */
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:20' deg = int32(0); */
    *deg = 0;

    /* 'polyfit_lhf_surf_point:21' prcurvs = [0;0]; */
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:21' maxprdir = [0;0;0]; */
    for (i = 0; i < 3; i++) {
      maxprdir[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      /* 'polyfit_lhf_surf_point:23' elseif nverts>=MAXNPNTS */
      /* 'polyfit_lhf_surf_point:24' nverts = MAXNPNTS-1; */
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    /* 'polyfit_lhf_surf_point:28' nrm = nrms_coor(v,1:3)'; */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    /* 'polyfit_lhf_surf_point:29' absnrm = abs(nrm); */
    b_abs(nrm, absnrm);

    /* 'polyfit_lhf_surf_point:31' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      /* 'polyfit_lhf_surf_point:32' t1 = [0; 1; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv6[i];
      }
    } else {
      /* 'polyfit_lhf_surf_point:33' else */
      /* 'polyfit_lhf_surf_point:34' t1 = [1; 0; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv7[i];
      }
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 - t1' * nrm * nrm; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * nrm[iy];
      b_ix++;
      iy++;
    }

    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] -= y * nrm[ix];
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 / sqrt(t1'*t1); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * absnrm[iy];
      b_ix++;
      iy++;
    }

    y = sqrt(y);
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] /= y;
    }

    emxInit_real_T(&us, 2);
    b_emxInit_real_T(&bs, 1);
    b_emxInit_real_T(&ws_row, 1);

    /* 'polyfit_lhf_surf_point:38' t2 = cross_col( nrm, t1); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    /* 'polyfit_lhf_surf_point:41' us = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:42' bs = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:43' ws_row = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:45' us(1,:)=0; */
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:45' ws_row(1)=1; */
    ws_row->data[0] = 1.0;

    /* 'polyfit_lhf_surf_point:46' for ii=1:nverts */
    for (i = 0; i + 1 <= nverts; i++) {
      /* 'polyfit_lhf_surf_point:47' u = xs(ngbvs(ii),1:3)-xs(v,1:3); */
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      /* 'polyfit_lhf_surf_point:49' us(ii+1-int32(interp),1) = u*t1; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;

      /* 'polyfit_lhf_surf_point:50' us(ii+1-int32(interp),2) = u*t2; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;

      /* 'polyfit_lhf_surf_point:51' bs(ii+1-int32(interp)) = u*nrm; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * nrm[iy];
        b_ix++;
        iy++;
      }

      bs->data[i] = y;

      /*  Compute normal-based weights */
      /* 'polyfit_lhf_surf_point:54' ws_row(ii+1-int32(interp)) = max(0, nrms_coor(ngbvs(ii),:)*nrm); */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += nrms_coor->data[(ngbvs[i] + nrms_coor->size[0] * b_ix) - 1] *
          nrm[iy];
        b_ix++;
        iy++;
      }

      y = 0.0 >= y ? 0.0 : y;
      ws_row->data[i] = y;
    }

    /* 'polyfit_lhf_surf_point:57' if degree==0 */
    if (degree == 0) {
      /*  Use linear fitting without weight */
      /* 'polyfit_lhf_surf_point:59' ws_row(:) = 1; */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      /* 'polyfit_lhf_surf_point:59' degree=int32(1); */
      degree = 1;
    }

    /*  Compute the coefficients */
    /* 'polyfit_lhf_surf_point:63' [bs, deg] = eval_vander_bivar( us, bs, degree, ws_row, interp, guardosc); */
    *deg = degree;
    eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    /* 'polyfit_lhf_surf_point:66' if deg<=1 */
    /* 'polyfit_lhf_surf_point:67' coder.varsize('cs', [6,1],[1,0]); */
    /* 'polyfit_lhf_surf_point:68' cs = bs(2-int32(interp):n); */
    /* 'polyfit_lhf_surf_point:70' grad = [cs(1); cs(2)]; */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];

    /* 'polyfit_lhf_surf_point:71' nrm_l = [-grad; 1]/sqrt(1+grad'*grad); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      y += grad[b_ix] * grad[iy];
      b_ix++;
      iy++;
    }

    y = sqrt(1.0 + y);
    for (i = 0; i < 2; i++) {
      nrm_l[i] = -grad[i] / y;
    }

    nrm_l[2] = 1.0 / y;

    /* 'polyfit_lhf_surf_point:73' P = [t1, t2, nrm]; */
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
    /* 'polyfit_lhf_surf_point:75' nrm = [P(1,:) * nrm_l; P(2,:) * nrm_l; P(3,:) * nrm_l]; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    b_y = 0.0;
    i = 0;
    b_iy = 0;
    c_y = 0.0;
    ix = 0;
    c_iy = 0;
    for (k = 0; k < 3; k++) {
      y += P[3 * b_ix] * nrm_l[iy];
      b_ix++;
      iy++;
      b_y += P[1 + 3 * i] * nrm_l[b_iy];
      i++;
      b_iy++;
      c_y += P[2 + 3 * ix] * nrm_l[c_iy];
      ix++;
      c_iy++;
    }

    nrm[0] = y;
    nrm[1] = b_y;
    nrm[2] = c_y;

    /* 'polyfit_lhf_surf_point:77' if deg>1 */
    if (*deg > 1) {
      /* 'polyfit_lhf_surf_point:78' H = [2*cs(3) cs(4); cs(4) 2*cs(5)]; */
      H[0] = 2.0 * bs->data[2];
      H[2] = bs->data[3];
      H[1] = bs->data[3];
      H[3] = 2.0 * bs->data[4];
    } else if (nverts >= 2) {
      /* 'polyfit_lhf_surf_point:79' elseif deg<=1 && nverts>=2 */
      /* 'polyfit_lhf_surf_point:80' if deg==0 && nverts>=2 */
      if (*deg == 0) {
        emxInit_real_T(&b_us, 2);

        /* 'polyfit_lhf_surf_point:81' us = us(1:3-int32(interp),:); */
        ix = b_us->size[0] * b_us->size[1];
        b_us->size[0] = 2;
        b_us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)b_us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          for (b_iy = 0; b_iy < 2; b_iy++) {
            b_us->data[b_iy + b_us->size[0] * ix] = us->data[b_iy + us->size[0] *
              ix];
          }
        }

        ix = us->size[0] * us->size[1];
        us->size[0] = b_us->size[0];
        us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          i = b_us->size[0] - 1;
          for (b_iy = 0; b_iy <= i; b_iy++) {
            us->data[b_iy + us->size[0] * ix] = b_us->data[b_iy + b_us->size[0] *
              ix];
          }
        }

        emxFree_real_T(&b_us);

        /* 'polyfit_lhf_surf_point:82' ws_row(1:3-int32(interp)) = 1; */
        for (ix = 0; ix < 2; ix++) {
          ws_row->data[ix] = 1.0;
        }
      }

      /*  Try to compute curvatures from normals */
      /* 'polyfit_lhf_surf_point:86' cs2 = linfit_lhf_grad_surf_point( ngbvs, us, t1, t2, nrms_coor, ws_row, interp); */
      linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row, cs2);

      /* 'polyfit_lhf_surf_point:87' H = [cs2(1) cs2(2); cs2(2) cs2(3)]; */
      H[0] = cs2[0];
      H[2] = cs2[1];
      H[1] = cs2[1];
      H[3] = cs2[2];
    } else {
      /* 'polyfit_lhf_surf_point:88' else */
      /* 'polyfit_lhf_surf_point:89' H = coder.nullcopy(zeros(2,2)); */
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);

    /* 'polyfit_lhf_surf_point:92' if deg>=1 */
    if (*deg >= 1) {
      /* 'polyfit_lhf_surf_point:93' if nargout==3 */
      /* 'polyfit_lhf_surf_point:95' else */
      /* 'polyfit_lhf_surf_point:96' [prcurvs, maxprdir_l] = eval_curvature_lhf_surf(grad, H); */
      eval_curvature_lhf_surf(grad, H, prcurvs, absnrm);

      /*  maxprdir = P * maxprdir_l; */
      /* 'polyfit_lhf_surf_point:98' maxprdir = [P(1,:) * maxprdir_l; P(2,:) * maxprdir_l; P(3,:) * maxprdir_l]; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      b_y = 0.0;
      i = 0;
      b_iy = 0;
      c_y = 0.0;
      ix = 0;
      c_iy = 0;
      for (k = 0; k < 3; k++) {
        y += P[3 * b_ix] * absnrm[iy];
        b_ix++;
        iy++;
        b_y += P[1 + 3 * i] * absnrm[b_iy];
        i++;
        b_iy++;
        c_y += P[2 + 3 * ix] * absnrm[c_iy];
        ix++;
        c_iy++;
      }

      maxprdir[0] = y;
      maxprdir[1] = b_y;
      maxprdir[2] = c_y;
    } else {
      /* 'polyfit_lhf_surf_point:100' else */
      /* 'polyfit_lhf_surf_point:101' prcurvs = [0;0]; */
      for (i = 0; i < 2; i++) {
        prcurvs[i] = 0.0;
      }

      /* 'polyfit_lhf_surf_point:102' maxprdir = [0;0;0]; */
      for (i = 0; i < 3; i++) {
        maxprdir[i] = 0.0;
      }
    }
  }
}

/*
 * function bs = compute_qtb( Q, bs, ncols)
 */
static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols)
{
  int32_T nrow;
  int32_T k;
  int32_T jj;
  real_T t2;
  int32_T ii;

  /* 'compute_qtb:3' nrow = int32(size(Q,1)); */
  nrow = Q->size[0];

  /* 'compute_qtb:4' for k=1:ncols */
  for (k = 0; k + 1 <= ncols; k++) {
    /*  Optimized version for */
    /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
    /*  where v is Q(k:npngs) */
    /* 'compute_qtb:8' for jj=1:int32(size(bs,2)) */
    for (jj = 0; jj < 2; jj++) {
      /* 'compute_qtb:9' t2 = 0; */
      t2 = 0.0;

      /* 'compute_qtb:10' for ii=k:nrow */
      for (ii = k; ii + 1 <= nrow; ii++) {
        /* 'compute_qtb:10' t2 = t2+Q(ii,k)*bs(ii,jj); */
        t2 += Q->data[ii + Q->size[0] * k] * bs->data[ii + bs->size[0] * jj];
      }

      /* 'compute_qtb:11' t2 = t2+t2; */
      t2 += t2;

      /* 'compute_qtb:12' for ii=k:nrow */
      for (ii = k; ii + 1 <= nrow; ii++) {
        /* 'compute_qtb:12' bs(ii,jj) = bs(ii,jj) - t2 * Q(ii,k); */
        bs->data[ii + bs->size[0] * jj] -= t2 * Q->data[ii + Q->size[0] * k];
      }
    }
  }
}

/*
 * function [curvs, dir, Jinv] = eval_curvature_lhf_surf( grad, H)
 */
static void eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3])
{
  real_T grad_sqnorm;
  real_T y;
  real_T tmp;
  real_T grad_norm;
  real_T ell;
  real_T c;
  real_T s;
  real_T v[2];
  real_T d1[2];
  int32_T ix;
  int32_T iy;
  int32_T k;
  real_T a[2];
  real_T W[4];
  real_T U[6];
  int32_T b_ix;
  int32_T b_iy;
  int32_T c_ix;
  int32_T c_iy;

  /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
  /* and pseudo-inverse. */
  /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
  /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
  /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
  /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
  /*  height function in 2x2 H with a local coordinate frame. */
  /*  */
  /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
  /* 'eval_curvature_lhf_surf:12' grad_sqnorm = grad(1)^2+grad(2)^2; */
  grad_sqnorm = grad[0];
  y = pow(grad_sqnorm, 2.0);
  grad_sqnorm = grad[1];
  tmp = pow(grad_sqnorm, 2.0);
  grad_sqnorm = y + tmp;

  /* 'eval_curvature_lhf_surf:13' grad_norm = sqrt(grad_sqnorm); */
  grad_norm = sqrt(grad_sqnorm);

  /*  Compute key parameters */
  /* 'eval_curvature_lhf_surf:16' ell = sqrt(1+grad_sqnorm); */
  ell = sqrt(1.0 + grad_sqnorm);

  /* 'eval_curvature_lhf_surf:17' ell2=1+grad_sqnorm; */
  /* 'eval_curvature_lhf_surf:17' ell3 = ell*(1+grad_sqnorm); */
  /* 'eval_curvature_lhf_surf:18' if grad_norm==0 */
  if (grad_norm == 0.0) {
    /* 'eval_curvature_lhf_surf:19' c = 1; */
    c = 1.0;

    /* 'eval_curvature_lhf_surf:19' s=0; */
    s = 0.0;
  } else {
    /* 'eval_curvature_lhf_surf:20' else */
    /* 'eval_curvature_lhf_surf:21' c = grad(1)/grad_norm; */
    c = grad[0] / grad_norm;

    /* 'eval_curvature_lhf_surf:21' s = grad(2)/grad_norm; */
    s = grad[1] / grad_norm;
  }

  /*  Compute mean curvature and Gaussian curvature */
  /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
  /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
  /*  Solve quadratic equation to compute principal curvatures */
  /* 'eval_curvature_lhf_surf:29' v = [c*H(1,1)+s*H(1,2) c*H(1,2)+s*H(2,2)]; */
  v[0] = c * H[0] + s * H[2];
  v[1] = c * H[2] + s * H[3];

  /* 'eval_curvature_lhf_surf:30' W1 = [v*[c; s]/ell3, v*[-s; c]/ell2]; */
  d1[0] = c;
  d1[1] = s;
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    y += v[ix] * d1[iy];
    ix++;
    iy++;
  }

  d1[0] = -s;
  d1[1] = c;
  tmp = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    tmp += v[ix] * d1[iy];
    ix++;
    iy++;
  }

  v[0] = y / (ell * (1.0 + grad_sqnorm));
  v[1] = tmp / (1.0 + grad_sqnorm);

  /* 'eval_curvature_lhf_surf:31' W = [W1; W1(2) [c*H(1,2)-s*H(1,1), c*H(2,2)-s*H(1,2)]*[-s; c]/ell]; */
  a[0] = c * H[2] - s * H[0];
  a[1] = c * H[3] - s * H[2];
  d1[0] = -s;
  d1[1] = c;
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 2; k++) {
    y += a[ix] * d1[iy];
    ix++;
    iy++;
    W[k << 1] = v[k];
  }

  W[1] = v[1];
  W[3] = y / ell;

  /*  Lambda = eig(W); */
  /* 'eval_curvature_lhf_surf:34' kH2 = W(1,1)+W(2,2); */
  grad_sqnorm = W[0] + W[3];

  /* 'eval_curvature_lhf_surf:35' tmp = sqrt((W(1,1)-W(2,2))*(W(1,1)-W(2,2))+4*W(1,2)*W(1,2)); */
  tmp = sqrt((W[0] - W[3]) * (W[0] - W[3]) + 4.0 * W[2] * W[2]);

  /* 'eval_curvature_lhf_surf:36' if kH2>0 */
  if (grad_sqnorm > 0.0) {
    /* 'eval_curvature_lhf_surf:37' curvs = 0.5*[kH2+tmp; kH2-tmp]; */
    curvs[0] = 0.5 * (grad_sqnorm + tmp);
    curvs[1] = 0.5 * (grad_sqnorm - tmp);
  } else {
    /* 'eval_curvature_lhf_surf:38' else */
    /* 'eval_curvature_lhf_surf:39' curvs = 0.5*[kH2-tmp; kH2+tmp]; */
    curvs[0] = 0.5 * (grad_sqnorm - tmp);
    curvs[1] = 0.5 * (grad_sqnorm + tmp);
  }

  /* 'eval_curvature_lhf_surf:42' if nargout > 1 */
  /*  Compute principal directions, first with basis of left  */
  /*  singular vectors of Jacobian */
  /*  Compute principal directions in 3-D space */
  /* 'eval_curvature_lhf_surf:47' U = [c/ell -s; s/ell c; grad_norm/ell 0]; */
  U[0] = c / ell;
  U[3] = -s;
  U[1] = s / ell;
  U[4] = c;
  U[2] = grad_norm / ell;
  U[5] = 0.0;

  /* 'eval_curvature_lhf_surf:49' if curvs(1)==curvs(2) */
  if (curvs[0] == curvs[1]) {
    /* 'eval_curvature_lhf_surf:50' dir = U(:,1); */
    for (ix = 0; ix < 3; ix++) {
      dir[ix] = U[ix];
    }
  } else {
    /* 'eval_curvature_lhf_surf:51' else */
    /* 'eval_curvature_lhf_surf:52' if abs(W(1,1)-curvs(2))>abs(W(1,1)-curvs(1)) */
    if (fabs(W[0] - curvs[1]) > fabs(W[0] - curvs[0])) {
      /* 'eval_curvature_lhf_surf:53' d1 = [W(1,1)-curvs(2); W(1,2)]; */
      d1[0] = W[0] - curvs[1];
      d1[1] = W[2];
    } else {
      /* 'eval_curvature_lhf_surf:54' else */
      /* 'eval_curvature_lhf_surf:55' d1 = [-W(1,2); W(1,1)-curvs(1)]; */
      d1[0] = -W[2];
      d1[1] = W[0] - curvs[0];
    }

    /* 'eval_curvature_lhf_surf:58' d1 = d1/sqrt(d1'*d1); */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      y += d1[ix] * d1[iy];
      ix++;
      iy++;
    }

    grad_sqnorm = sqrt(y);
    for (ix = 0; ix < 2; ix++) {
      d1[ix] /= grad_sqnorm;
    }

    /* 'eval_curvature_lhf_surf:59' dir = [U(1,:)*d1; U(2,:)*d1; U(3,:)*d1]; */
    y = 0.0;
    ix = 0;
    iy = 0;
    tmp = 0.0;
    b_ix = 0;
    b_iy = 0;
    grad_sqnorm = 0.0;
    c_ix = 0;
    c_iy = 0;
    for (k = 0; k < 2; k++) {
      y += U[3 * ix] * d1[iy];
      ix++;
      iy++;
      tmp += U[1 + 3 * b_ix] * d1[b_iy];
      b_ix++;
      b_iy++;
      grad_sqnorm += U[2 + 3 * c_ix] * d1[c_iy];
      c_ix++;
      c_iy++;
    }

    dir[0] = y;
    dir[1] = tmp;
    dir[2] = grad_sqnorm;
  }

  /* 'eval_curvature_lhf_surf:62' if nargout>2 */
}

/*
 * function [bs, degree] = eval_vander_bivar(us, bs, degree, ws, interp0, guardosc)
 */
static void eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws)
{
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T nrow;
  int32_T k;
  emxArray_real_T *b_V;
  int32_T c_V;
  int32_T i3;
  int32_T loop_ub;
  int32_T b_loop_ub;
  emxArray_real_T *ws1;
  real_T t2;
  int32_T b_bs[2];
  int32_T c_bs[2];
  emxArray_real_T d_bs;
  emxArray_real_T e_bs;
  int32_T f_bs[2];
  int32_T g_bs[2];
  emxArray_real_T *D;
  int32_T exitg1;
  int32_T h_bs[2];
  int32_T i_bs[2];
  int32_T j_bs[2];
  int32_T k_bs[2];
  int32_T l_bs[2];
  int32_T m_bs[2];
  emxArray_real_T n_bs;
  int32_T o_bs[2];
  int32_T p_bs[2];
  int32_T q_bs[2];
  int32_T r_bs[2];

  /* EVAL_VANDER_BIVAR Evaluate generalized Vandermonde matrix. */
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC) */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*  */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /* 'eval_vander_bivar:10' coder.extrinsic('save') */
  /* 'eval_vander_bivar:11' degree = int32(degree); */
  /* 'eval_vander_bivar:12' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:15' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:16' if nargin<5 */
  /* 'eval_vander_bivar:17' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:20' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  ncols = (*degree + 2) * (*degree + 1) / 2 - 1;

  /* 'eval_vander_bivar:21' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (*degree > 1)) {
    /* 'eval_vander_bivar:22' degree=degree-1; */
    (*degree)--;

    /* 'eval_vander_bivar:23' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
    ncols = (*degree + 2) * (*degree + 1) / 2 - 1;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  /* 'eval_vander_bivar:27' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, *degree, V);

  /* 'eval_vander_bivar:28' if interp0 */
  /* 'eval_vander_bivar:28' V=V(:,2:end); */
  nrow = V->size[1];
  if (2 > nrow) {
    k = 0;
    nrow = 0;
  } else {
    k = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i3 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = nrow - k;
  emxEnsureCapacity((emxArray__common *)b_V, i3, (int32_T)sizeof(real_T));
  loop_ub = (nrow - k) - 1;
  for (nrow = 0; nrow <= loop_ub; nrow++) {
    b_loop_ub = c_V - 1;
    for (i3 = 0; i3 <= b_loop_ub; i3++) {
      b_V->data[i3 + b_V->size[0] * nrow] = V->data[i3 + V->size[0] * (k + nrow)];
    }
  }

  nrow = V->size[0] * V->size[1];
  V->size[0] = b_V->size[0];
  V->size[1] = b_V->size[1];
  emxEnsureCapacity((emxArray__common *)V, nrow, (int32_T)sizeof(real_T));
  loop_ub = b_V->size[1] - 1;
  for (nrow = 0; nrow <= loop_ub; nrow++) {
    b_loop_ub = b_V->size[0] - 1;
    for (k = 0; k <= b_loop_ub; k++) {
      V->data[k + V->size[0] * nrow] = b_V->data[k + b_V->size[0] * nrow];
    }
  }

  emxFree_real_T(&b_V);

  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar:31' if nargin>3 && ~isempty(ws) */
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:32' if degree>2 */
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      /* 'eval_vander_bivar:33' ws1 = us(:,1).*us(:,1)+us(:,2).*us(:,2); */
      nrow = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (nrow = 0; nrow <= loop_ub; nrow++) {
        ws1->data[nrow] = us->data[nrow] * us->data[nrow] + us->data[nrow +
          us->size[0]] * us->data[nrow + us->size[0]];
      }

      /* 'eval_vander_bivar:34' ws1 = ws1 + sum(ws1)/double(npnts)*1.e-2; */
      t2 = sum(ws1);
      t2 = t2 / (real_T)npnts * 0.01;
      nrow = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (nrow = 0; nrow <= loop_ub; nrow++) {
        ws1->data[nrow] += t2;
      }

      /* 'eval_vander_bivar:35' if degree<4 */
      if (*degree < 4) {
        /* 'eval_vander_bivar:36' for ii=1:npnts */
        for (c_V = 0; c_V + 1 <= npnts; c_V++) {
          /* 'eval_vander_bivar:37' if ws1(ii)~=0 */
          if (ws1->data[c_V] != 0.0) {
            /* 'eval_vander_bivar:38' ws1(ii) = ws(ii) / sqrt(ws1(ii)); */
            ws1->data[c_V] = ws->data[c_V] / sqrt(ws1->data[c_V]);
          } else {
            /* 'eval_vander_bivar:39' else */
            /* 'eval_vander_bivar:40' ws1(ii) = ws(ii); */
            ws1->data[c_V] = ws->data[c_V];
          }
        }
      } else {
        /* 'eval_vander_bivar:43' else */
        /* 'eval_vander_bivar:44' for ii=1:npnts */
        for (c_V = 0; c_V + 1 <= npnts; c_V++) {
          /* 'eval_vander_bivar:45' if ws1(ii)~=0 */
          if (ws1->data[c_V] != 0.0) {
            /* 'eval_vander_bivar:46' ws1(ii) = ws(ii) / ws1(ii); */
            ws1->data[c_V] = ws->data[c_V] / ws1->data[c_V];
          } else {
            /* 'eval_vander_bivar:47' else */
            /* 'eval_vander_bivar:48' ws1(ii) = ws(ii); */
            ws1->data[c_V] = ws->data[c_V];
          }
        }
      }

      /* 'eval_vander_bivar:52' for ii=1:npnts */
      for (c_V = 0; c_V + 1 <= npnts; c_V++) {
        /* 'eval_vander_bivar:53' for jj=1:ncols */
        for (nrow = 0; nrow + 1 <= ncols; nrow++) {
          /* 'eval_vander_bivar:53' V(ii,jj) = V(ii,jj) * ws1(ii); */
          V->data[c_V + V->size[0] * nrow] *= ws1->data[c_V];
        }

        /* 'eval_vander_bivar:54' for jj=1:size(bs,2) */
        /* 'eval_vander_bivar:54' bs(ii,jj) = bs(ii,jj) * ws1(ii); */
        b_bs[0] = bs->size[0];
        b_bs[1] = 1;
        c_bs[0] = bs->size[0];
        c_bs[1] = 1;
        d_bs = *bs;
        d_bs.size = (int32_T *)&b_bs;
        d_bs.numDimensions = 1;
        e_bs = *bs;
        e_bs.size = (int32_T *)&c_bs;
        e_bs.numDimensions = 1;
        d_bs.data[c_V] = e_bs.data[c_V] * ws1->data[c_V];
      }
    } else {
      /* 'eval_vander_bivar:56' else */
      /* 'eval_vander_bivar:57' for ii=1:npnts */
      for (c_V = 0; c_V + 1 <= npnts; c_V++) {
        /* 'eval_vander_bivar:58' for jj=1:ncols */
        for (nrow = 0; nrow + 1 <= ncols; nrow++) {
          /* 'eval_vander_bivar:58' V(ii,jj) = V(ii,jj) * ws(ii); */
          V->data[c_V + V->size[0] * nrow] *= ws->data[c_V];
        }

        /* 'eval_vander_bivar:59' for jj=1:int32(size(bs,2)) */
        /* 'eval_vander_bivar:59' bs(ii,jj) = bs(ii,jj) * ws(ii); */
        f_bs[0] = bs->size[0];
        f_bs[1] = 1;
        g_bs[0] = bs->size[0];
        g_bs[1] = 1;
        d_bs = *bs;
        d_bs.size = (int32_T *)&f_bs;
        d_bs.numDimensions = 1;
        e_bs = *bs;
        e_bs.size = (int32_T *)&g_bs;
        e_bs.numDimensions = 1;
        d_bs.data[c_V] = e_bs.data[c_V] * ws->data[c_V];
      }
    }
  }

  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:66' ts = coder.nullcopy(zeros(ncols,1)); */
  nrow = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:67' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:70' D = coder.nullcopy(zeros(ncols,1)); */
  nrow = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, nrow, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:71' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  nrow = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:74' ncols_sub = ncols; */
  /* 'eval_vander_bivar:75' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (nrow < ncols) {
      /* 'eval_vander_bivar:76' degree = degree-1; */
      (*degree)--;

      /* 'eval_vander_bivar:78' if degree==0 */
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar:80' bs(:) = 0; */
        nrow = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, nrow, (int32_T)sizeof(real_T));
        loop_ub = bs->size[0] - 1;
        for (nrow = 0; nrow <= loop_ub; nrow++) {
          bs->data[nrow] = 0.0;
        }

        exitg1 = 1U;
      } else {
        /* 'eval_vander_bivar:82' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U) - 1;
      }
    } else {
      /* % Compute Q'bs */
      /* 'eval_vander_bivar:86' bs = compute_qtb( V, bs, ncols_sub); */
      /* 'compute_qtb:3' nrow = int32(size(Q,1)); */
      nrow = V->size[0];

      /* 'compute_qtb:4' for k=1:ncols */
      for (k = 0; k + 1 <= ncols; k++) {
        /*  Optimized version for */
        /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
        /*  where v is Q(k:npngs) */
        /* 'compute_qtb:8' for jj=1:int32(size(bs,2)) */
        /* 'compute_qtb:9' t2 = 0; */
        t2 = 0.0;

        /* 'compute_qtb:10' for ii=k:nrow */
        for (c_V = k; c_V + 1 <= nrow; c_V++) {
          /* 'compute_qtb:10' t2 = t2+Q(ii,k)*bs(ii,jj); */
          h_bs[0] = bs->size[0];
          h_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&h_bs;
          d_bs.numDimensions = 1;
          t2 += V->data[c_V + V->size[0] * k] * d_bs.data[c_V];
        }

        /* 'compute_qtb:11' t2 = t2+t2; */
        t2 += t2;

        /* 'compute_qtb:12' for ii=k:nrow */
        for (c_V = k; c_V + 1 <= nrow; c_V++) {
          /* 'compute_qtb:12' bs(ii,jj) = bs(ii,jj) - t2 * Q(ii,k); */
          i_bs[0] = bs->size[0];
          i_bs[1] = 1;
          j_bs[0] = bs->size[0];
          j_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&i_bs;
          d_bs.numDimensions = 1;
          e_bs = *bs;
          e_bs.size = (int32_T *)&j_bs;
          e_bs.numDimensions = 1;
          d_bs.data[c_V] = e_bs.data[c_V] - t2 * V->data[c_V + V->size[0] * k];
        }
      }

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar:89' for i=1:ncols_sub */
      for (nrow = 0; nrow + 1 <= ncols; nrow++) {
        /* 'eval_vander_bivar:89' V(i,i) = D(i); */
        V->data[nrow + V->size[0] * nrow] = D->data[nrow];
      }

      /* 'eval_vander_bivar:90' if guardosc */
      /* 'eval_vander_bivar:92' else */
      /* 'eval_vander_bivar:93' bs = backsolve(V, bs, ncols_sub, ts); */
      /*  Perform backward substitution. */
      /*      bs = backsolve(R, bs) */
      /*      bs = backsolve(R, bs, cend) */
      /*      bs = backsolve(R, bs, cend, ws) */
      /*   Compute bs = (triu(R(1:cend,1:cend))\bs) ./ ws; */
      /*   The right-hand side vector bs can have multiple columns. */
      /*   By default, cend is size(R,2), and ws is ones. */
      /* 'backsolve:10' if nargin<3 */
      /* 'backsolve:12' for kk=1:int32(size(bs,2)) */
      /* 'backsolve:13' for jj=cend:-1:1 */
      for (nrow = ncols - 1; nrow + 1 > 0; nrow--) {
        /* 'backsolve:14' for ii=jj+1:cend */
        for (c_V = nrow + 1; c_V + 1 <= ncols; c_V++) {
          /* 'backsolve:15' bs(jj,kk) = bs(jj,kk) - R(jj,ii) * bs(ii,kk); */
          k_bs[0] = bs->size[0];
          k_bs[1] = 1;
          l_bs[0] = bs->size[0];
          l_bs[1] = 1;
          m_bs[0] = bs->size[0];
          m_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&k_bs;
          d_bs.numDimensions = 1;
          e_bs = *bs;
          e_bs.size = (int32_T *)&l_bs;
          e_bs.numDimensions = 1;
          n_bs = *bs;
          n_bs.size = (int32_T *)&m_bs;
          n_bs.numDimensions = 1;
          d_bs.data[nrow] = e_bs.data[nrow] - V->data[nrow + V->size[0] * c_V] *
            n_bs.data[c_V];
        }

        /* 'backsolve:18' assert( R(jj,jj)~=0); */
        /* 'backsolve:19' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
        o_bs[0] = bs->size[0];
        o_bs[1] = 1;
        p_bs[0] = bs->size[0];
        p_bs[1] = 1;
        d_bs = *bs;
        d_bs.size = (int32_T *)&o_bs;
        d_bs.numDimensions = 1;
        e_bs = *bs;
        e_bs.size = (int32_T *)&p_bs;
        e_bs.numDimensions = 1;
        d_bs.data[nrow] = e_bs.data[nrow] / V->data[nrow + V->size[0] * nrow];
      }

      /* 'backsolve:23' if nargin>3 */
      /*  Scale bs back if ts is given. */
      /* 'backsolve:25' for kk=1:int32(size(bs,2)) */
      /* 'backsolve:26' for jj = 1:cend */
      for (nrow = 0; nrow + 1 <= ncols; nrow++) {
        /* 'backsolve:27' bs(jj,kk) = bs(jj,kk) / ws(jj); */
        q_bs[0] = bs->size[0];
        q_bs[1] = 1;
        r_bs[0] = bs->size[0];
        r_bs[1] = 1;
        d_bs = *bs;
        d_bs.size = (int32_T *)&q_bs;
        d_bs.numDimensions = 1;
        e_bs = *bs;
        e_bs.size = (int32_T *)&r_bs;
        e_bs.numDimensions = 1;
        d_bs.data[nrow] = e_bs.data[nrow] / ws1->data[nrow];
      }

      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
}

/*
 * function [V, ords] = gen_vander_bivar(us, degree, cols, dderiv, rows)
 */
static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V)
{
  int32_T npnts;
  emxArray_real_T *b_us;
  int32_T ncols;
  int32_T i0;
  int32_T c;
  emxArray_real_T *v1;
  emxArray_real_T *c_us;
  emxArray_real_T *v2;
  int32_T p;
  emxArray_real_T *r0;
  emxArray_real_T *y;
  emxArray_real_T *a;
  emxArray_real_T *b_v2;
  int32_T nx;
  int32_T kk2;
  int32_T sz[2];
  static const int8_T iv2[10] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55 };

  emxArray_int32_T *r1;
  int8_T iv3[2];

  /*  Construct generalized Vandermonde matrix for two independent variables, */
  /*    using both function values and derivatives. */
  /*  */
  /*  V = gen_vander_bivar(us, degree, [], dderiv): us specifies local */
  /*    coordinates of points, degree specifies the degree of polynomial, and */
  /*    dderiv specifies the highest degree of derivative available (the */
  /*    default value is 0). When degree>0, columns are ordered based on */
  /*    Pascal triangle. When degree<=0, columns are ordered based on */
  /*    Pascal quadrilateral. Rows are always ordered based on Pascal triangle. */
  /*    For example, for degree==2 and dderiv==1, V looks as follows: */
  /*             1, u1, v1, u1^2, u1*v1, v1^2 */
  /*             1, u2, v2, u2^2, u2*v2, v2^2 */
  /*             ... */
  /*             0, 1,  0,  2u1,  v1,    0 */
  /*             0, 1,  0,  2u2,  v2,    0 */
  /*             ... */
  /*             0, 0,  1,  0,    u1,  2v1 */
  /*             0, 0,  1,  0,    u2,  2v2 */
  /*             ... */
  /*  */
  /*  V = gen_vander_bivar(us, degree, [], dderiv, rows) obtains only */
  /*    selected rows for each point as specified by rows. */
  /*  */
  /*  V = gen_vander_bivar(us, degree, cols, dderiv, rows) obtains only selected */
  /*    rows and columns in generalized Vandermonde matrix. */
  /*  */
  /*  [V, ords] = gen_vander_bivar(...) also returns the order of derivatives */
  /*    for each row, where ords is (size(V,1)/size(us,1))-by-1. */
  /*  */
  /*  See also gen_vander_univar, gen_vander_trivar */
  /* 'gen_vander_bivar:33' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'gen_vander_bivar:34' assert( size(us,2)==2); */
  /* 'gen_vander_bivar:35' sz2 = int32([1  3  6  10  15  21  28  36   45   55]); */
  /* 'gen_vander_bivar:37' if nargin<2 */
  /* 'gen_vander_bivar:38' if nargin<3 */
  /* 'gen_vander_bivar:38' cols = zeros(0,1,'int32'); */
  /* 'gen_vander_bivar:39' if nargin<4 */
  /* 'gen_vander_bivar:39' dderiv = int32(0); */
  /* 'gen_vander_bivar:40' if nargin<5 */
  /* 'gen_vander_bivar:40' rows = zeros(0,1,'int32'); */
  /* 'gen_vander_bivar:48' assert(dderiv<length(sz2) && degree<length(sz2)); */
  /* 'gen_vander_bivar:49' nrpp = sz2( dderiv+1); */
  /* 'gen_vander_bivar:51' if degree<=0 */
  if (degree <= 0) {
    b_emxInit_real_T(&b_us, 1);

    /* 'gen_vander_bivar:52' degree = -degree; */
    degree = -degree;

    /* 'gen_vander_bivar:53' ncols = int32( (1+degree)*(1+degree)); */
    ncols = (1 + degree) * (1 + degree);

    /* 'gen_vander_bivar:54' nrows = npnts*nrpp; */
    /* 'gen_vander_bivar:56' if isempty(coder.target) && isequal(class(us),'sym') */
    /* 'gen_vander_bivar:58' else */
    /* 'gen_vander_bivar:59' V = coder.nullcopy(zeros(nrows, ncols, class(us))); */
    i0 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = ncols;
    emxEnsureCapacity((emxArray__common *)V, i0, (int32_T)sizeof(real_T));

    /*  Preallocate storage */
    /*  Use tensor product */
    /* 'gen_vander_bivar:63' v1 = gen_vander_univar(us(:,1), degree, [], dderiv); */
    i0 = b_us->size[0];
    b_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)b_us, i0, (int32_T)sizeof(real_T));
    c = us->size[0] - 1;
    for (i0 = 0; i0 <= c; i0++) {
      b_us->data[i0] = us->data[i0];
    }

    emxInit_real_T(&v1, 2);
    b_emxInit_real_T(&c_us, 1);
    gen_vander_univar(b_us, degree, v1);

    /* 'gen_vander_bivar:64' v2 = gen_vander_univar(us(:,2), degree, [], dderiv); */
    i0 = c_us->size[0];
    c_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)c_us, i0, (int32_T)sizeof(real_T));
    emxFree_real_T(&b_us);
    c = us->size[0] - 1;
    for (i0 = 0; i0 <= c; i0++) {
      c_us->data[i0] = us->data[i0 + us->size[0]];
    }

    emxInit_real_T(&v2, 2);
    gen_vander_univar(c_us, degree, v2);

    /* 'gen_vander_bivar:66' for p=1:npnts */
    p = 0;
    emxFree_real_T(&c_us);
    emxInit_real_T(&r0, 2);
    emxInit_real_T(&y, 2);
    b_emxInit_real_T(&a, 1);
    emxInit_real_T(&b_v2, 2);
    while (p + 1 <= npnts) {
      /* 'gen_vander_bivar:67' V(p,:) = reshape(v1(p,:)'*v2(p,:),1,ncols); */
      i0 = a->size[0];
      a->size[0] = v1->size[1];
      emxEnsureCapacity((emxArray__common *)a, i0, (int32_T)sizeof(real_T));
      c = v1->size[1] - 1;
      for (i0 = 0; i0 <= c; i0++) {
        a->data[i0] = v1->data[p + v1->size[0] * i0];
      }

      i0 = b_v2->size[0] * b_v2->size[1];
      b_v2->size[0] = 1;
      b_v2->size[1] = v2->size[1];
      emxEnsureCapacity((emxArray__common *)b_v2, i0, (int32_T)sizeof(real_T));
      c = v2->size[1] - 1;
      for (i0 = 0; i0 <= c; i0++) {
        b_v2->data[b_v2->size[0] * i0] = v2->data[p + v2->size[0] * i0];
      }

      i0 = y->size[0] * y->size[1];
      y->size[0] = a->size[0];
      y->size[1] = b_v2->size[1];
      emxEnsureCapacity((emxArray__common *)y, i0, (int32_T)sizeof(real_T));
      c = b_v2->size[1] - 1;
      for (i0 = 0; i0 <= c; i0++) {
        nx = a->size[0] - 1;
        for (kk2 = 0; kk2 <= nx; kk2++) {
          y->data[kk2 + y->size[0] * i0] = a->data[kk2] * b_v2->data[b_v2->size
            [0] * i0];
        }
      }

      nx = y->size[0] * y->size[1];
      for (i0 = 0; i0 < 2; i0++) {
        sz[i0] = 0;
      }

      sz[0] = 1;
      sz[1] = ncols;
      i0 = r0->size[0] * r0->size[1];
      r0->size[0] = 1;
      r0->size[1] = sz[1];
      emxEnsureCapacity((emxArray__common *)r0, i0, (int32_T)sizeof(real_T));
      for (c = 0; c + 1 <= nx; c++) {
        r0->data[c] = y->data[c];
      }

      c = r0->size[1] - 1;
      for (i0 = 0; i0 <= c; i0++) {
        V->data[p + V->size[0] * i0] = r0->data[r0->size[0] * i0];
      }

      p++;
    }

    emxFree_real_T(&b_v2);
    emxFree_real_T(&a);
    emxFree_real_T(&y);
    emxFree_real_T(&r0);
    emxFree_real_T(&v2);
    emxFree_real_T(&v1);

    /* 'gen_vander_bivar:70' r = int32(npnts); */
    /* 'gen_vander_bivar:71' for deg=1:dderiv */
  } else {
    /* 'gen_vander_bivar:80' else */
    /* 'gen_vander_bivar:81' ncols = sz2( degree+1); */
    /* 'gen_vander_bivar:83' if isempty(coder.target) && isequal(class(us),'sym') */
    /* 'gen_vander_bivar:85' else */
    /* 'gen_vander_bivar:86' V = zeros(npnts*nrpp, ncols, class(us)); */
    i0 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = (int32_T)iv2[degree];
    emxEnsureCapacity((emxArray__common *)V, i0, (int32_T)sizeof(real_T));
    c = npnts * iv2[degree] - 1;
    for (i0 = 0; i0 <= c; i0++) {
      V->data[i0] = 0.0;
    }

    /*  Preallocate storage */
    /*     %% Compute rows corresponding to function values */
    /* 'gen_vander_bivar:90' V(1:npnts,1) = 1; */
    if (1 > npnts) {
      i0 = 0;
    } else {
      i0 = npnts;
    }

    b_emxInit_int32_T(&r1, 1);
    kk2 = r1->size[0];
    r1->size[0] = i0;
    emxEnsureCapacity((emxArray__common *)r1, kk2, (int32_T)sizeof(int32_T));
    c = i0 - 1;
    for (i0 = 0; i0 <= c; i0++) {
      r1->data[i0] = 1 + i0;
    }

    c = r1->size[0];
    emxFree_int32_T(&r1);
    c--;
    for (i0 = 0; i0 <= c; i0++) {
      V->data[i0] = 1.0;
    }

    /* 'gen_vander_bivar:91' V(1:npnts,2:3) = us; */
    for (i0 = 0; i0 < 2; i0++) {
      iv3[i0] = (int8_T)(i0 + 1);
    }

    for (i0 = 0; i0 < 2; i0++) {
      c = us->size[0] - 1;
      for (kk2 = 0; kk2 <= c; kk2++) {
        V->data[kk2 + V->size[0] * iv3[i0]] = us->data[kk2 + us->size[0] * i0];
      }
    }

    /* 'gen_vander_bivar:93' c = int32(4); */
    c = 3;

    /* 'gen_vander_bivar:94' for kk=2:degree */
    for (nx = 2; nx <= degree; nx++) {
      /* 'gen_vander_bivar:95' for kk2=1:kk */
      for (kk2 = 1; kk2 <= nx; kk2++) {
        /* 'gen_vander_bivar:96' for p=1:npnts */
        for (p = 0; p + 1 <= npnts; p++) {
          /* 'gen_vander_bivar:96' V(p,c) = V(p,c-kk)*us(p,1); */
          V->data[p + V->size[0] * c] = V->data[p + V->size[0] * (c - nx)] *
            us->data[p];
        }

        /* 'gen_vander_bivar:97' c = c + 1; */
        c++;
      }

      /* 'gen_vander_bivar:100' for p=1:npnts */
      for (p = 0; p + 1 <= npnts; p++) {
        /* 'gen_vander_bivar:100' V(p,c) = V(p,c-kk-1)*us(p,2); */
        V->data[p + V->size[0] * c] = V->data[p + V->size[0] * ((c - nx) - 1)] *
          us->data[p + us->size[0]];
      }

      /* 'gen_vander_bivar:101' c = c + 1; */
      c++;
    }

    /*     %% Add rows corresponding to derivatives */
    /* 'gen_vander_bivar:105' r = int32(npnts); */
    /* 'gen_vander_bivar:106' fact_degd = int32(1); */
    /* 'gen_vander_bivar:108' for degd=1:min(dderiv,degree) */
  }

  /* 'gen_vander_bivar:154' V = subvander( V, npnts, rows, cols); */
  /*  Select subset of Vandermond matrix. */
  /*      V = subvander( V, npnts, rows, cols) */
  /* 'subvander:5' if ~isempty(rows) */
  /* 'gen_vander_bivar:156' if nargout>1 */
}

/*
 * function [V, ords] = gen_vander_univar(us, degree, cols, dderiv, rows)
 */
static void gen_vander_univar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V)
{
  int32_T npnts;
  int32_T ncols;
  int32_T p;
  int32_T loop_ub;
  emxArray_int32_T *r2;

  /*  Construct generalized Vandermonde matrix for one independent variable, */
  /*     using function values and optionally the derivatives. */
  /*  */
  /*  V = gen_vander_univar(us, degree, [], dderiv); us specifies local */
  /*    coordinates of points, degree specifies the degree of polynomial, */
  /*    and dderiv specifies the highest degree of derivative available (the */
  /*    default value is 0). For dderiv==1, V looks as follows: */
  /*             1, u1, u1^2, u1^3, u1^4, ... */
  /*             1, u2, u2^2, u2^3, u2^4, ... */
  /*             ... */
  /*             0, 1, 2u1, 3u1^2, 4u1^3, ... */
  /*             0, 1, 2u2, 3u2^2, 4u2^3, ... */
  /*             ... */
  /*  */
  /*  V = gen_vander_univar(us, degree, [], dderiv, rows) obtains only */
  /*    selected rows for each point as specified by rows. */
  /*  */
  /*  V = gen_vander_univar(us, degree, cols, dderiv, rows) obtains only */
  /*    selected rows and selected columns in the matrix. */
  /*  */
  /*  [V, ords] = gen_vander_univar(...) also returns the order of derivatives */
  /*    for each row, where ords is (size(V,1)/size(us,1))-by-1. */
  /*  */
  /*  See also gen_vander_bivar, gen_vander_trivar */
  /* 'gen_vander_univar:27' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'gen_vander_univar:28' assert( size(us,2)==1); */
  /* 'gen_vander_univar:30' if nargin<2 */
  /* 'gen_vander_univar:31' if nargin<3 */
  /* 'gen_vander_univar:32' if nargin<4 */
  /* 'gen_vander_univar:33' else */
  /* 'gen_vander_univar:33' dderiv = int32(dderiv); */
  /* 'gen_vander_univar:34' if nargin<5 */
  /* 'gen_vander_univar:34' rows = zeros(0,1,'int32'); */
  /* 'gen_vander_univar:39' degree = abs(degree); */
  if (degree < 0) {
    degree = -degree;
  }

  /* 'gen_vander_univar:40' ncols = degree+1; */
  ncols = degree + 1;

  /* 'gen_vander_univar:41' coder.varsize('V', [inf,inf]); */
  /* 'gen_vander_univar:42' if isempty(coder.target) && isequal( class(us), 'sym') */
  /* 'gen_vander_univar:44' else */
  /* 'gen_vander_univar:45' V = zeros(npnts*(dderiv+1), ncols, class(us)); */
  p = V->size[0] * V->size[1];
  V->size[0] = npnts;
  V->size[1] = ncols;
  emxEnsureCapacity((emxArray__common *)V, p, (int32_T)sizeof(real_T));
  loop_ub = npnts * ncols - 1;
  for (p = 0; p <= loop_ub; p++) {
    V->data[p] = 0.0;
  }

  /*  Preallocate storage */
  /* % Compute rows corresponding to function values */
  /* 'gen_vander_univar:49' V(1:npnts,1) = 1; */
  if (1 > npnts) {
    p = 0;
  } else {
    p = npnts;
  }

  b_emxInit_int32_T(&r2, 1);
  ncols = r2->size[0];
  r2->size[0] = p;
  emxEnsureCapacity((emxArray__common *)r2, ncols, (int32_T)sizeof(int32_T));
  loop_ub = p - 1;
  for (p = 0; p <= loop_ub; p++) {
    r2->data[p] = 1 + p;
  }

  ncols = r2->size[0];
  emxFree_int32_T(&r2);
  loop_ub = ncols - 1;
  for (p = 0; p <= loop_ub; p++) {
    V->data[p] = 1.0;
  }

  /* 'gen_vander_univar:51' if degree>0 */
  if (degree > 0) {
    /* 'gen_vander_univar:52' V(1:npnts,2) = us(:); */
    ncols = us->size[0];
    loop_ub = ncols - 1;
    for (p = 0; p <= loop_ub; p++) {
      V->data[p + V->size[0]] = us->data[p];
    }

    /* 'gen_vander_univar:54' for ii=2:degree+1 */
    loop_ub = degree + 1;
    for (ncols = 1; ncols + 1 <= loop_ub; ncols++) {
      /* 'gen_vander_univar:55' for p=1:npnts */
      for (p = 0; p + 1 <= npnts; p++) {
        /* 'gen_vander_univar:56' V(p,ii)=V(p,ii-1)*us(p); */
        V->data[p + V->size[0] * ncols] = V->data[p + V->size[0] * (ncols - 1)] *
          us->data[p];
      }
    }
  }

  /* % Add rows corresponding to the derivatives multiplied by corresponding power of u */
  /* 'gen_vander_univar:62' fact_k = 1; */
  /* 'gen_vander_univar:63' r = int32(npnts); */
  /* 'gen_vander_univar:65' for k=1:min(dderiv,degree) */
  /* 'gen_vander_univar:79' V = subvander( V, npnts, rows, cols); */
  /*  Select subset of Vandermond matrix. */
  /*      V = subvander( V, npnts, rows, cols) */
  /* 'subvander:5' if ~isempty(rows) */
  /* 'gen_vander_univar:81' if nargout>1 */
}

/*
 * function [hess, deg] = linfit_lhf_grad_surf_point( ngbvs, us, t1, t2, nrms, ws, interp)
 */
static void linfit_lhf_grad_surf_point(const int32_T ngbvs[128], const
  emxArray_real_T *us, const real_T t1[3], const real_T t2[3], const
  emxArray_real_T *nrms, const emxArray_real_T *ws, real_T hess[3])
{
  emxArray_real_T *bs;
  int32_T loop_ub;
  int32_T ii;
  real_T b;
  int32_T ix;
  int32_T iy;
  int32_T k;
  emxInit_real_T(&bs, 2);

  /*  Computes principal curvatures and principal direction from normals. */
  /*  This function is invoked only if there are insufficient points in the stencil. */
  /* 'polyfit_lhf_surf_point:109' bs = coder.nullcopy(zeros( size(us,1),2)); */
  loop_ub = bs->size[0] * bs->size[1];
  bs->size[0] = us->size[0];
  bs->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)bs, loop_ub, (int32_T)sizeof(real_T));

  /* 'polyfit_lhf_surf_point:111' for ii=1:int32(size(us,1)) - 1 + int32(interp) */
  loop_ub = us->size[0];
  for (ii = 0; ii + 1 <= loop_ub; ii++) {
    /* 'polyfit_lhf_surf_point:112' nrm_ii = nrms(ngbvs(ii),1:3); */
    /* 'polyfit_lhf_surf_point:113' w = ws(ii+1-int32(interp)); */
    /* 'polyfit_lhf_surf_point:115' if w>0 */
    if (ws->data[ii] > 0.0) {
      /* 'polyfit_lhf_surf_point:116' bs(ii+1-int32(interp),1) = -(nrm_ii*t1)/w; */
      b = 0.0;
      ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        b += nrms->data[(ngbvs[ii] + nrms->size[0] * ix) - 1] * t1[iy];
        ix++;
        iy++;
      }

      bs->data[ii] = -b / ws->data[ii];

      /* 'polyfit_lhf_surf_point:117' bs(ii+1-int32(interp),2) = -(nrm_ii*t2)/w; */
      b = 0.0;
      ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        b += nrms->data[(ngbvs[ii] + nrms->size[0] * ix) - 1] * t2[iy];
        ix++;
        iy++;
      }

      bs->data[ii + bs->size[0]] = -b / ws->data[ii];
    }
  }

  /*  Compute the coefficients and store them */
  /* 'polyfit_lhf_surf_point:122' [bs, deg] = eval_vander_bivar( us, bs, int32(1), ws, interp, false); */
  b_eval_vander_bivar(us, bs, ws);

  /* 'polyfit_lhf_surf_point:123' hess = [bs(2-int32(interp),1) 0.5*(bs(3-int32(interp),1)+bs(2-int32(interp),2)) bs(3-int32(interp),2)]; */
  b = bs->data[1] + bs->data[bs->size[0]];
  hess[0] = bs->data[0];
  hess[1] = 0.5 * b;
  hess[2] = bs->data[1 + bs->size[0]];
  emxFree_real_T(&bs);
}

/*
 * function s = norm2_vec( v, dim)
 */
static real_T norm2_vec(const emxArray_real_T *v)
{
  real_T s;
  real_T w;
  uint32_T ii;
  real_T u0;

  /* NORM2_VEC Computes the 2-norm of a vector. */
  /*  NORM2_VEC(V) Computes the 2-norm of a row or column vector V. */
  /*  NORM2_VEC(V,dim) If dim==1, computes the 2-norm of column vectors of V, */
  /*        If dim==2, computes the 2-norm of row vectors of V. */
  /*  */
  /*  See also SQNORM2_VEC */
  /*  */
  /*  Note: This routine uses rescaling to guard against overflow/underflow. */
  /*  It does not inline. Use sqrt( sqnorm2_vec(v)) to produce more efficient */
  /*  code that can be inlined but does not perform rescaling. */
  /* 'norm2_vec:13' coder.inline('never'); */
  /* 'norm2_vec:15' assert(nargin==1 || dim==1 || dim==2); */
  /* 'norm2_vec:17' if nargin==1 */
  /* 'norm2_vec:18' w = cast(0, class(v)); */
  w = 0.0;

  /* 'norm2_vec:19' for ii=1:numel(v) */
  for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
    /* 'norm2_vec:19' w = max(w,abs(v(ii))); */
    u0 = fabs(v->data[(int32_T)ii - 1]);
    w = w >= u0 ? w : u0;
  }

  /* 'norm2_vec:21' s = cast(0, class(v)); */
  s = 0.0;

  /* 'norm2_vec:22' if w==0 */
  if (w == 0.0) {
    /*  W can be zero for max(0,nan,...). Adding all three entries */
    /*  together will make sure NaN will be preserved. */
    /* 'norm2_vec:25' for ii=1:numel(v) */
    for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
      /* 'norm2_vec:25' s = s + v(ii); */
      s += v->data[(int32_T)ii - 1];
    }
  } else {
    /* 'norm2_vec:26' else */
    /* 'norm2_vec:27' for ii=1:numel(v) */
    for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
      /* 'norm2_vec:27' s = s + (v(ii)/w)^2; */
      u0 = v->data[(int32_T)ii - 1] / w;
      u0 = pow(u0, 2.0);
      s += u0;
    }

    /* 'norm2_vec:29' s = w*sqrt(s); */
    s = w * sqrt(s);
  }

  return s;
}

/*
 * function [ngbvs, nverts, vtags, ftags, ngbfs, nfaces] = obtain_nring_surf...
 *     ( vid, ring, minpnts, tris, opphes, v2he, ngbvs, vtags, ftags, ngbfs)
 */
static int32_T c_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags, const int32_T ngbfs[256])
{
  int32_T nverts;
  int32_T b_ngbfs[256];
  int32_T fid;
  int32_T lid;
  int32_T nfaces;
  boolean_T overflow;
  boolean_T b2;
  int32_T fid_in;
  static const int8_T iv16[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv17[3] = { 3, 1, 2 };

  int32_T opp;
  int32_T nverts_pre;
  int32_T nfaces_pre;
  real_T ring_full;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T b3;
  boolean_T isfirst;
  int32_T exitg3;
  boolean_T guard2 = FALSE;
  memcpy((void *)&b_ngbfs[0], (void *)&ngbfs[0], sizeof(int32_T) << 8);

  /* OBTAIN_NRING_SURF Collect n-ring vertices and faces of a triangle mesh. */
  /*  [NGBVS,NVERTS,VTAGS,FTAGS,NGBFS,NFACES] = OBTAIN_NRING_SURF(VID,RING, ... */
  /*  MINPNTS,TRIS,OPPHES,V2HE,NGBVS,VTAGS,FTAGS,NGBFS)  Collects n-ring */
  /*  vertices and faces of a vertex and saves them into NGBVS and NGBFS, */
  /*  where n is a floating point number with 0.5 increments (1, 1.5, 2, etc.) */
  /*  We define the n-ring verticse as follows: */
  /*   - 0-ring: vertex itself */
  /*   - k-ring vertices: vertices that share an edge with (k-1)-ring vertices */
  /*   - (k+0.5)-ring vertices: k-ring plus vertices that share an element */
  /*            with two vertices of k-ring vertices. */
  /*  For triangle meshes, the k-ring vertices always form some triangles. */
  /*  */
  /*  Input arguments */
  /*    vid: vertex ID */
  /*    ring: the desired number of rings (it is a float as it can have halves) */
  /*    minpnts: the minimum number of points desired */
  /*    tris: element connectivity */
  /*    opphes: opposite half-edges */
  /*    v2he: vertex-to-halfedge mapping */
  /*    ngbvs: buffer space for neighboring vertices (not including vid itself) */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*  */
  /*  Output arguments */
  /*    ngbvs: buffer space for neighboring vertices */
  /*    nverts: number of vertices in the neighborhood */
  /*    vtags: vertex tags (boolean, of length equal to number of vertices) */
  /*    ftags: face tags (boolean, of length equal to number of elements) */
  /*    ngbfs: buffer space for neighboring faces */
  /*    nfaces: number of elements in the neighborhood */
  /*  */
  /*  Notes */
  /*   1. vtags and ftags must be set to false at input. They are reset to */
  /*      false at output. */
  /*   2. Since the vertex itself is always in ring, we do not include it in */
  /*      the output array ngbvs. */
  /*   3. If NGBVS or NGBFS is not enough to store the whole neighborhood, */
  /*      then only a subset of the neighborhood will be returned. */
  /*      The maximum number of points returned is numel(NGBVS) if NGBVS is */
  /*      given as part of the input, or 128 if not an input arguement. */
  /*      The maximum number of faces returned is numel(NGBFS) if NGBFS is */
  /*      given as part of the input, or 256 if not an input arguement. */
  /*  */
  /*  See also OBTAIN_NRING_SURF, OBTAIN_NRING_QUAD, OBTAIN_NRING_CURV, OBTAIN_NRING_VOL */
  /* 'obtain_nring_surf:49' coder.extrinsic('warning'); */
  /*  assert(isscalar(vid)&&isa(vid,'int32')); */
  /*  assert(isscalar(ring)&&isa(ring,'double')); */
  /*  assert(isscalar(minpnts)&&isa(minpnts,'int32')); */
  /*  assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /*  assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /*  assert((size(ngbvs,2)==1) && (size(ngbvs,1)>=1) && isa(ngbvs,'int32')); */
  /*  assert((size(vtags,2)==1) && (size(vtags,1)>=1) && isa(vtags,'logical')); */
  /*  assert((size(ftags,2)==1) && (size(ftags,1)>=1) && isa(ftags,'logical')); */
  /*  assert((size(ngbfs,2)==1) && (size(ngbfs,1)>=1) && isa(ngbfs,'int32')); */
  /* 'obtain_nring_surf:61' MAXNPNTS = int32(128); */
  /* 'obtain_nring_surf:63' assert(ring>=1 && floor(ring*2)==ring*2); */
  /* 'obtain_nring_surf:64' if nargin>=8 */
  /* 'obtain_nring_surf:64' assert( islogical( vtags)); */
  /* 'obtain_nring_surf:65' if nargin>=9 */
  /* 'obtain_nring_surf:65' assert( islogical(ftags)); */
  /* 'obtain_nring_surf:67' fid = heid2fid(v2he(vid)); */
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  /* 'heid2fid:3' coder.inline('always'); */
  /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
  fid = (int32_T)((uint32_T)v2he->data[vid - 1] >> 2U) - 1;

  /* 'obtain_nring_surf:67' lid = heid2leid(v2he(vid)); */
  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  /* 'heid2leid:3' coder.inline('always'); */
  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
  lid = (int32_T)((uint32_T)v2he->data[vid - 1] & 3U);

  /* 'obtain_nring_surf:68' nverts=int32(0); */
  nverts = 0;

  /* 'obtain_nring_surf:68' nfaces=int32(0); */
  nfaces = 0;

  /* 'obtain_nring_surf:68' overflow = false; */
  overflow = FALSE;

  /* 'obtain_nring_surf:70' if ~fid */
  if (!(fid + 1 != 0)) {
  } else {
    /* 'obtain_nring_surf:72' prv = int32([3 1 2]); */
    /* 'obtain_nring_surf:73' nxt = int32([2 3 1]); */
    /* 'obtain_nring_surf:75' if nargin>=7 && ~isempty(ngbvs) */
    /* 'obtain_nring_surf:76' maxnv = int32(numel(ngbvs)); */
    /* 'obtain_nring_surf:81' if nargin>=10 && ~isempty(ngbfs) */
    /* 'obtain_nring_surf:82' maxnf = int32(numel(ngbfs)); */
    /* 'obtain_nring_surf:87' oneringonly = ring==1 && minpnts==0 && nargout<5; */
    if ((ring == 1.0) && (minpnts == 0)) {
      b2 = TRUE;
    } else {
      b2 = FALSE;
    }

    /* 'obtain_nring_surf:88' hebuf = coder.nullcopy(zeros(maxnv,1, 'int32')); */
    /*  Optimized version for collecting one-ring vertices */
    /* 'obtain_nring_surf:91' if opphes( fid, lid) */
    if (opphes->data[fid + opphes->size[0] * lid] != 0) {
      /* 'obtain_nring_surf:92' fid_in = fid; */
      fid_in = fid + 1;
    } else {
      /* 'obtain_nring_surf:93' else */
      /* 'obtain_nring_surf:94' fid_in = int32(0); */
      fid_in = 0;

      /* 'obtain_nring_surf:96' v = tris(fid, nxt(lid)); */
      /* 'obtain_nring_surf:97' nverts = int32(1); */
      nverts = 1;

      /* 'obtain_nring_surf:97' ngbvs( 1) = v; */
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv16[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      if (!b2) {
        /* 'obtain_nring_surf:99' hebuf(1) = 0; */
        hebuf[0] = 0;
      }
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    /* 'obtain_nring_surf:103' while 1 */
    do {
      exitg4 = 0U;

      /*  Insert vertx into list */
      /* 'obtain_nring_surf:105' lid_prv = prv(lid); */
      lid = iv17[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        if (!b2) {
          /*  Save starting position for next vertex */
          /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv17[lid] -
            1)];

          /* 'obtain_nring_surf:114' nfaces = nfaces + 1; */
          nfaces++;

          /* 'obtain_nring_surf:114' ngbfs( nfaces) = fid; */
          b_ngbfs[nfaces - 1] = fid + 1;
        }
      } else {
        /* 'obtain_nring_surf:116' else */
        /* 'obtain_nring_surf:117' overflow = true; */
        overflow = TRUE;
      }

      /* 'obtain_nring_surf:120' opp = opphes(fid, lid_prv); */
      opp = opphes->data[fid + opphes->size[0] * lid];

      /* 'obtain_nring_surf:121' fid = heid2fid(opp); */
      /*  HEID2FID   Obtains face ID from half-edge ID. */
      /* 'heid2fid:3' coder.inline('always'); */
      /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
      fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] * lid] >> 2U)
        - 1;

      /* 'obtain_nring_surf:123' if fid == fid_in */
      if (fid + 1 == fid_in) {
        exitg4 = 1U;
      } else {
        /* 'obtain_nring_surf:125' else */
        /* 'obtain_nring_surf:126' lid = heid2leid(opp); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        lid = (int32_T)((uint32_T)opp & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    /* 'obtain_nring_surf:130' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
    if ((ring == 1.0) && ((nverts >= minpnts) || (nverts >= 128) || (nfaces >=
          256))) {
      /* 'obtain_nring_surf:131' if overflow */
    } else {
      /* 'obtain_nring_surf:137' vtags(vid) = true; */
      vtags->data[vid - 1] = TRUE;

      /* 'obtain_nring_surf:138' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_surf:138' vtags(ngbvs(i))=true; */
        vtags->data[ngbvs[lid - 1] - 1] = TRUE;
      }

      /* 'obtain_nring_surf:139' for i=1:nfaces */
      for (lid = 1; lid <= nfaces; lid++) {
        /* 'obtain_nring_surf:139' ftags(ngbfs(i))=true; */
        ftags->data[b_ngbfs[lid - 1] - 1] = TRUE;
      }

      /*  Define buffers and prepare tags for further processing */
      /* 'obtain_nring_surf:142' nverts_pre = int32(0); */
      nverts_pre = 0;

      /* 'obtain_nring_surf:143' nfaces_pre = int32(0); */
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      /* 'obtain_nring_surf:146' ring_full = fix( ring); */
      ring_full = ring;
      b_fix(&ring_full);

      /* 'obtain_nring_surf:147' minpnts = min(minpnts, maxnv); */
      minpnts = minpnts <= 128 ? minpnts : 128;

      /* 'obtain_nring_surf:149' cur_ring=1; */
      cur_ring = 1.0;

      /* 'obtain_nring_surf:150' while true */
      do {
        exitg1 = 0U;

        /* 'obtain_nring_surf:151' if cur_ring>ring_full || (cur_ring==ring_full && ring_full~=ring) */
        guard1 = FALSE;
        if ((cur_ring > ring_full) || ((cur_ring == ring_full) && (ring_full !=
              ring))) {
          /*  Collect halfring */
          /* 'obtain_nring_surf:153' nfaces_last = nfaces; */
          opp = nfaces;

          /* 'obtain_nring_surf:153' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_surf:154' for ii = nfaces_pre+1 : nfaces_last */
          while (nfaces_pre + 1 <= opp) {
            /*  take opposite vertex in opposite face */
            /* 'obtain_nring_surf:156' for jj=int32(1):3 */
            lid = 0;
            exitg2 = 0U;
            while ((exitg2 == 0U) && (lid + 1 < 4)) {
              /* 'obtain_nring_surf:157' oppe = opphes( ngbfs(ii), jj); */
              /* 'obtain_nring_surf:158' fid = heid2fid(oppe); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)opphes->data[(b_ngbfs[nfaces_pre] +
                opphes->size[0] * lid) - 1] >> 2U) - 1;

              /* 'obtain_nring_surf:160' if oppe && ~ftags(fid) */
              if ((opphes->data[(b_ngbfs[nfaces_pre] + opphes->size[0] * lid) -
                   1] != 0) && (!ftags->data[fid])) {
                /* 'obtain_nring_surf:161' lid = heid2leid(oppe); */
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                /* 'heid2leid:3' coder.inline('always'); */
                /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                lid = (int32_T)((uint32_T)opphes->data[(b_ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);

                /* 'obtain_nring_surf:162' v = tris( fid, prv(lid)); */
                /* 'obtain_nring_surf:164' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:165'                         ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  (iv17[lid] - 1)] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:166' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:167' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:167' ngbfs( nfaces) = fid; */
                  b_ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:168' ftags(fid) = true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:171' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv17[lid] -
                      1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv17[lid]
                    - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  vtags->data[tris->data[fid + tris->size[0] * (iv17[lid] - 1)]
                    - 1] = TRUE;
                }

                exitg2 = 1U;
              } else {
                lid++;
              }
            }

            nfaces_pre++;
          }

          /* 'obtain_nring_surf:180' if nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nfaces==nfaces_last */
          if ((nverts >= minpnts) || (nfaces >= 256) || (nfaces == opp)) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:182' else */
            /*  If needs to expand, then undo the last half ring */
            /* 'obtain_nring_surf:184' for i=nverts_last+1:nverts */
            for (lid = nverts_last; lid + 1 <= nverts; lid++) {
              /* 'obtain_nring_surf:184' vtags(ngbvs(i)) = false; */
              vtags->data[ngbvs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:185' nverts = nverts_last; */
            nverts = nverts_last;

            /* 'obtain_nring_surf:187' for i=nfaces_last+1:nfaces */
            for (lid = opp; lid + 1 <= nfaces; lid++) {
              /* 'obtain_nring_surf:187' ftags(ngbfs(i)) = false; */
              ftags->data[b_ngbfs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:188' nfaces = nfaces_last; */
            nfaces = opp;
            guard1 = TRUE;
          }
        } else {
          guard1 = TRUE;
        }

        if (guard1 == TRUE) {
          /*  Collect next full level of ring */
          /* 'obtain_nring_surf:193' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_surf:193' nfaces_pre = nfaces; */
          nfaces_pre = nfaces;

          /* 'obtain_nring_surf:194' for ii=nverts_pre+1 : nverts_last */
          while (nverts_pre + 1 <= nverts_last) {
            /* 'obtain_nring_surf:195' v = ngbvs(ii); */
            /* 'obtain_nring_surf:195' fid = heid2fid(v2he(v)); */
            /*  HEID2FID   Obtains face ID from half-edge ID. */
            /* 'heid2fid:3' coder.inline('always'); */
            /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
            fid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] >> 2U) -
              1;

            /* 'obtain_nring_surf:195' lid = heid2leid(v2he(v)); */
            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            /* 'heid2leid:3' coder.inline('always'); */
            /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
            lid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] & 3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            /* 'obtain_nring_surf:199' allow_early_term = hebuf(ii) && opphes(fid,lid); */
            if ((hebuf[nverts_pre] != 0) && (opphes->data[fid + opphes->size[0] *
                 lid] != 0)) {
              b3 = TRUE;
            } else {
              b3 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b3) {
              /* 'obtain_nring_surf:201' fid = heid2fid(hebuf(ii)); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)hebuf[nverts_pre] >> 2U) - 1;

              /* 'obtain_nring_surf:201' lid = heid2leid(hebuf(ii)); */
              /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
              /* 'heid2leid:3' coder.inline('always'); */
              /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
              lid = (int32_T)((uint32_T)hebuf[nverts_pre] & 3U);
            }

            /*  */
            /* 'obtain_nring_surf:205' if opphes( fid, lid) */
            if (opphes->data[fid + opphes->size[0] * lid] != 0) {
              /* 'obtain_nring_surf:206' fid_in = fid; */
              fid_in = fid + 1;
            } else {
              /* 'obtain_nring_surf:207' else */
              /* 'obtain_nring_surf:208' fid_in = cast(0,class(fid)); */
              fid_in = 0;

              /* 'obtain_nring_surf:210' v = tris(fid, nxt(lid)); */
              /* 'obtain_nring_surf:211' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs); */
              if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                (iv16[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                nverts++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv16[lid]
                  - 1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                vtags->data[tris->data[fid + tris->size[0] * (iv16[lid] - 1)] -
                  1] = TRUE;

                /*  Save starting position for next vertex */
                /* 'obtain_nring_surf:215' hebuf(nverts) = 0; */
                hebuf[nverts - 1] = 0;
              }
            }

            /*  Rotate counterclockwise around the vertex. */
            /* 'obtain_nring_surf:220' isfirst=true; */
            isfirst = TRUE;

            /* 'obtain_nring_surf:221' while true */
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              /* 'obtain_nring_surf:223' lid_prv = prv(lid); */
              lid = iv17[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b3 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /* 'obtain_nring_surf:228' else */
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                /* 'obtain_nring_surf:231' v = tris(fid, lid_prv); */
                /* 'obtain_nring_surf:232' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:233'                     ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  lid] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:235' if ~vtags(v) && ~overflow */
                if ((!vtags->data[tris->data[fid + tris->size[0] * lid] - 1]) &&
                    (!overflow)) {
                  /* 'obtain_nring_surf:236' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:236' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

                  /* 'obtain_nring_surf:236' vtags(v)=true; */
                  vtags->data[tris->data[fid + tris->size[0] * lid] - 1] = TRUE;

                  /*  Save starting position for next ring */
                  /* 'obtain_nring_surf:239' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
                  hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv17[lid] - 1)];
                }

                /* 'obtain_nring_surf:242' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:243' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:243' ngbfs( nfaces) = fid; */
                  b_ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:243' ftags(fid)=true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:245' isfirst = false; */
                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                /* 'obtain_nring_surf:248' opp = opphes(fid, lid_prv); */
                opp = opphes->data[fid + opphes->size[0] * lid];

                /* 'obtain_nring_surf:249' fid = heid2fid(opp); */
                /*  HEID2FID   Obtains face ID from half-edge ID. */
                /* 'heid2fid:3' coder.inline('always'); */
                /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid] >> 2U) - 1;

                /* 'obtain_nring_surf:251' if fid == fid_in */
                if (fid + 1 == fid_in) {
                  /*  Finished cycle */
                  exitg3 = 1U;
                } else {
                  /* 'obtain_nring_surf:253' else */
                  /* 'obtain_nring_surf:254' lid = heid2leid(opp); */
                  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                  /* 'heid2leid:3' coder.inline('always'); */
                  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                  lid = (int32_T)((uint32_T)opp & 3U);
                }
              }
            } while (exitg3 == 0U);

            nverts_pre++;
          }

          /* 'obtain_nring_surf:259' cur_ring = cur_ring+1; */
          cur_ring++;

          /* 'obtain_nring_surf:260' if (nverts>=minpnts && cur_ring>=ring) || nfaces==nfaces_pre || overflow */
          if (((nverts >= minpnts) && (cur_ring >= ring)) || (nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:264' nverts_pre = nverts_last; */
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      /* 'obtain_nring_surf:268' vtags(vid) = false; */
      vtags->data[vid - 1] = FALSE;

      /* 'obtain_nring_surf:269' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_surf:269' vtags(ngbvs(i))=false; */
        vtags->data[ngbvs[lid - 1] - 1] = FALSE;
      }

      /* 'obtain_nring_surf:270' if ~oneringonly */
      if (!b2) {
        /* 'obtain_nring_surf:270' for i=1:nfaces */
        for (lid = 1; lid <= nfaces; lid++) {
          /* 'obtain_nring_surf:270' ftags(ngbfs(i))=false; */
          ftags->data[b_ngbfs[lid - 1] - 1] = FALSE;
        }
      }

      /* 'obtain_nring_surf:271' if overflow */
    }
  }

  return nverts;
}

/*
 * function [nrms,curs,prdirs] = polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, ...
 * nrms_proj, opphes, v2he, degree, ring, iterfit, interp, nrms, curs, prdirs)
 */
static void polyfit_lhf_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, const
  emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T
  ring, boolean_T iterfit, emxArray_real_T *nrms, emxArray_real_T *curs,
  emxArray_real_T *prdirs)
{
  static const int8_T iv14[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T nverts;
  int32_T minpntsv;
  emxArray_boolean_T *ftags;
  emxArray_int32_T *degs;
  boolean_T b0;
  boolean_T b1;
  int32_T ii;
  real_T ringv;
  int32_T exitg5;
  int32_T ngbfs[256];
  int32_T ngbvs[128];
  int32_T deg;
  real_T nrm[3];
  real_T prcurvs[2];
  real_T maxprdir[3];
  int32_T exitg4;
  static const int8_T iv15[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T exitg3;
  int32_T exitg2;
  int32_T exitg1;

  /* POLYFIT_LHF_SURF Compute polynomial fitting with adaptive reduced QR */
  /* factorization. */
  /*  [NRMS,CURS,PRDIRS] = POLYFIT_LHF_SURF(XS,TRIS,OPPHES,V2HE,DEGREE, ... */
  /*  RING,ITERFIT,INTERP,NRMS,CURS,PRDIRS) Computes polynomial fitting with */
  /*  adaptive reduced QR factorization using the following input and output */
  /*  arguments. */
  /*  Input:  XS:       nv*3 Coordinates of points */
  /*          TRIS:     matrix of size mx3 storing element connectivity */
  /*          OPPHES:   matrix of size mx3 storing opposite half edges */
  /*          V2HE:     incident halfedges of vertices */
  /*          DEGREE:   Degree of polynomials */
  /*          ITERFIT:  Whether or not to use iterative fitting */
  /*          INTERP:   Whether or not to enforce fitting to pass a given point. */
  /*  */
  /*  Output: NRMS:     Vertex norms (nx3) */
  /*          CURS:     Principal curvatures (nx2); */
  /*          PRDIRS:   Principal directions crt maximum curvature (nx3) */
  /*  */
  /*  See also POLYFIT_LHF_SURF_POINT */
  /* #   coder.typeof(int32(0),[inf,3],[1,0]), */
  /* #   coder.typeof(0,[inf,3],[1,0]), */
  /* #   coder.typeof(int32(0),[inf,3],[1,0]), */
  /* #   coder.typeof(int32(0),[inf,1],[1,0]), */
  /* #   int32(0),0,false,false, */
  /* #   coder.typeof(0,[inf,3],[1,0]), */
  /* #   coder.typeof(0,[inf,2],[1,0]), */
  /* #   coder.typeof(0,[inf,3],[1,0])} */
  /* 'polyfit_lhf_surf_cleanmesh:32' coder.extrinsic('fprintf','nonzeros','save') */
  /* 'polyfit_lhf_surf_cleanmesh:33' assert(isa(ring,'double')); */
  /*  ring is double, as we allow half rings. */
  /* 'polyfit_lhf_surf_cleanmesh:35' if nargout>=1 && nargin<11 */
  /* 'polyfit_lhf_surf_cleanmesh:36' if nargin<12 */
  /* 'polyfit_lhf_surf_cleanmesh:40' if nargin<13 */
  /* 'polyfit_lhf_surf_cleanmesh:45' if degree>0 && degree<=6 */
  if ((degree > 0) && (degree <= 6)) {
    /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
    /* 'polyfit_lhf_surf_cleanmesh:47' pntsneeded = [5 9 15 23 32 42]; */
    /* 'polyfit_lhf_surf_cleanmesh:48' minpnts = int32(pntsneeded(degree)); */
    minpnts = (int32_T)iv14[degree - 1];
  } else if (degree <= 0) {
    /* 'polyfit_lhf_surf_cleanmesh:49' elseif degree<=0 */
    /* 'polyfit_lhf_surf_cleanmesh:50' degree=int32(0); */
    degree = 0;

    /* 'polyfit_lhf_surf_cleanmesh:50' minpnts = int32(0); */
    minpnts = 0;

    /* 'polyfit_lhf_surf_cleanmesh:50' ring=1; */
    ring = 1.0;
  } else {
    /* 'polyfit_lhf_surf_cleanmesh:51' else */
    /* 'polyfit_lhf_surf_cleanmesh:52' degree=int32(6); */
    degree = 6;

    /* 'polyfit_lhf_surf_cleanmesh:52' minpnts = int32(0); */
    minpnts = 0;
  }

  emxInit_boolean_T(&vtags, 1);

  /* 'polyfit_lhf_surf_cleanmesh:55' MAXNPNTS=int32(128); */
  /* 'polyfit_lhf_surf_cleanmesh:56' nv = int32(size(xs, 1)); */
  nv = xs->size[0];

  /* 'polyfit_lhf_surf_cleanmesh:57' ngbvs = coder.nullcopy(zeros(MAXNPNTS,1, 'int32')); */
  /* 'polyfit_lhf_surf_cleanmesh:58' vtags = false(nv, 1); */
  nverts = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, nverts, (int32_T)sizeof(boolean_T));
  minpntsv = nv - 1;
  for (nverts = 0; nverts <= minpntsv; nverts++) {
    vtags->data[nverts] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);

  /* 'polyfit_lhf_surf_cleanmesh:59' ftags = false(size(tris,1), 1); */
  nverts = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, nverts, (int32_T)sizeof(boolean_T));
  minpntsv = tris->size[0] - 1;
  for (nverts = 0; nverts <= minpntsv; nverts++) {
    ftags->data[nverts] = FALSE;
  }

  b_emxInit_int32_T(&degs, 1);

  /* 'polyfit_lhf_surf_cleanmesh:60' degs = coder.nullcopy(zeros(nv,1, 'int32')); */
  nverts = degs->size[0];
  degs->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)degs, nverts, (int32_T)sizeof(int32_T));

  /* 'polyfit_lhf_surf_cleanmesh:62' noprdir = isempty(prdirs); */
  b0 = (prdirs->size[0] == 0);

  /* 'polyfit_lhf_surf_cleanmesh:63' normalonly = nargout==1 || degree==1 || (nargin>9 && isempty(curs)) || ... */
  /* 'polyfit_lhf_surf_cleanmesh:64'     (nargin>10 && isempty(curs) && isempty(prdirs)); */
  if ((degree == 1) || (curs->size[0] == 0) || ((curs->size[0] == 0) &&
       (prdirs->size[0] == 0))) {
    b1 = TRUE;
  } else {
    b1 = FALSE;
  }

  /* 'polyfit_lhf_surf_cleanmesh:66' for ii=1:nv_clean */
  for (ii = 1; ii <= nv_clean; ii++) {
    /* 'polyfit_lhf_surf_cleanmesh:67' ringv = ring; */
    ringv = ring;

    /* 'polyfit_lhf_surf_cleanmesh:69' while (1) */
    do {
      exitg5 = 0U;

      /*  Collect neighbor vertices */
      /* 'polyfit_lhf_surf_cleanmesh:71' maxnf = 2*MAXNPNTS; */
      /* 'polyfit_lhf_surf_cleanmesh:71' ngbfs = coder.nullcopy(zeros(maxnf,1, 'int32')); */
      /* 'polyfit_lhf_surf_cleanmesh:72' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpnts, ... */
      /* 'polyfit_lhf_surf_cleanmesh:73'             tris, opphes, v2he, ngbvs, vtags, ftags,ngbfs); */
      nverts = c_obtain_nring_surf(ii, ringv, minpnts, tris, opphes, v2he, ngbvs,
        vtags, ftags, ngbfs);

      /* 'polyfit_lhf_surf_cleanmesh:75' if normalonly */
      if (b1) {
        /* 'polyfit_lhf_surf_cleanmesh:76' [nrm, deg] = polyfit_lhf_surf_point( ii, ngbvs, nverts, xs, ... */
        /* 'polyfit_lhf_surf_cleanmesh:77'                 nrms_proj, degree, interp); */
        polyfit_lhf_surf_point(ii, ngbvs, nverts, xs, nrms_proj, degree, nrm,
          &deg);
      } else {
        /* 'polyfit_lhf_surf_cleanmesh:78' else */
        /* 'polyfit_lhf_surf_cleanmesh:79' if noprdir */
        if (b0) {
          /* 'polyfit_lhf_surf_cleanmesh:80' [nrm, deg, prcurvs] = polyfit_lhf_surf_point( ii, ngbvs, nverts, xs, ... */
          /* 'polyfit_lhf_surf_cleanmesh:81'                     nrms_proj, degree, interp); */
          b_polyfit_lhf_surf_point(ii, ngbvs, nverts, xs, nrms_proj, degree, nrm,
            &deg, prcurvs);
        } else {
          /* 'polyfit_lhf_surf_cleanmesh:82' else */
          /* 'polyfit_lhf_surf_cleanmesh:83' [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point( ii, ngbvs, nverts, xs, ... */
          /* 'polyfit_lhf_surf_cleanmesh:84'                     nrms_proj, degree, interp); */
          c_polyfit_lhf_surf_point(ii, ngbvs, nverts, xs, nrms_proj, degree, nrm,
            &deg, prcurvs, maxprdir);

          /* 'polyfit_lhf_surf_cleanmesh:86' if size(prdirs,1) */
          if (prdirs->size[0] != 0) {
            /* 'polyfit_lhf_surf_cleanmesh:86' prdirs(ii,1:3) = maxprdir'; */
            for (nverts = 0; nverts < 3; nverts++) {
              prdirs->data[(ii + prdirs->size[0] * nverts) - 1] =
                maxprdir[nverts];
            }
          }
        }

        /* 'polyfit_lhf_surf_cleanmesh:89' if size(curs,1) */
        if (curs->size[0] != 0) {
          /* 'polyfit_lhf_surf_cleanmesh:89' curs(ii,1:2) = prcurvs'; */
          for (nverts = 0; nverts < 2; nverts++) {
            curs->data[(ii + curs->size[0] * nverts) - 1] = prcurvs[nverts];
          }
        }
      }

      /* 'polyfit_lhf_surf_cleanmesh:91' degs(ii) = deg; */
      degs->data[ii - 1] = deg;

      /* 'polyfit_lhf_surf_cleanmesh:92' if size(nrms,1) */
      if (nrms->size[0] != 0) {
        /* 'polyfit_lhf_surf_cleanmesh:92' nrms(ii, 1:3) = nrm'; */
        for (nverts = 0; nverts < 3; nverts++) {
          nrms->data[(ii + nrms->size[0] * nverts) - 1] = nrm[nverts];
        }
      }

      /*  Enlarge the neighborhood if necessary */
      /* 'polyfit_lhf_surf_cleanmesh:95' if deg < degree && ringv<ring+ring */
      if ((deg < degree) && (ringv < ring + ring)) {
        /* 'polyfit_lhf_surf_cleanmesh:96' ringv=ringv+0.5; */
        ringv += 0.5;

        /*  Enlarge the neighborhood */
      } else {
        exitg5 = 1U;
      }
    } while (exitg5 == 0U);

    /* 'polyfit_lhf_surf_cleanmesh:97' else */
  }

  /* 'polyfit_lhf_surf_cleanmesh:103' if nargout==1 || (~iterfit && (nargout==2 && ~size(curs,1) || ... */
  /* 'polyfit_lhf_surf_cleanmesh:104'         (nargout==3 && ~size(curs,1) && ~size(prdirs,1)))) */
  if ((!iterfit) && (!(curs->size[0] != 0)) && (!(prdirs->size[0] != 0))) {
  } else {
    /* 'polyfit_lhf_surf_cleanmesh:108' assert(~isempty(degs)); */
    /* % */
    /* 'polyfit_lhf_surf_cleanmesh:110' if nargout==2 || isempty(prdirs) */
    if (prdirs->size[0] == 0) {
      /* 'polyfit_lhf_surf_cleanmesh:111' if iterfit */
      if (iterfit) {
        /* 'polyfit_lhf_surf_cleanmesh:112' curs = polyfit_lhfgrad_surf_cleanmesh(nv_clean, xs, nrms, tris, opphes, ... */
        /* 'polyfit_lhf_surf_cleanmesh:113'             v2he, degree, degree, ring, curs); */
        /* POLYFIT_LHFGRAD_SURF_CLEANMESH Compute polynomial fitting of gradients with adaptive */
        /* reduced QR factorization. */
        /*  [CURS,PRDIRS] = POLYFIT_LHFGRAD_SURF(XS,NRMS,TRIS,OPPHES,V2HE,DEGS, ... */
        /*  DEGREE,RING,CURS,PRDIRS) Computes polynomial fitting of gradients with  */
        /*  adaptive reduced QR factorization using the following input and output */
        /*  arguments. */
        /*  Input:  XS:       nv*3 Coordinates of points */
        /*          NRMS:     Normals to be fit */
        /*          TRIS:     matrix of size mx3 storing element connectivity */
        /*          OPPTR:    matrix of size mx3 storing opposite vertices */
        /*          DEGREE:   Degree of polynomials */
        /*          RECUR:    Whether or not to use iterative fitting */
        /*          STRIP:    Whether or not to enforce fitting to pass a given point. */
        /*   */
        /*  Output: CURS:     Principal curvatures (nx2); */
        /*          PRDIRS:   Principal directions crt maximum curvature (nx3) */
        /*  */
        /*  See also POLYFIT_LHFGRAD_SURF_POINT */
        /* 'polyfit_lhfgrad_surf_cleanmesh:21' MAXNPNTS=int32(128); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:22' assert(isa(ring,'double')); */
        /*  ring is double, as we allow half rings. */
        /* 'polyfit_lhfgrad_surf_cleanmesh:24' if degree<=6 */
        /* 'polyfit_lhfgrad_surf_cleanmesh:25' pntsneeded = int32([5 9 15 23 32 42]); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:26' minpnts = pntsneeded(degree); */
        /*  Compute fitting at all vertices */
        /* 'polyfit_lhfgrad_surf_cleanmesh:32' nv = int32(size(xs, 1)); */
        nv = xs->size[0];

        /* 'polyfit_lhfgrad_surf_cleanmesh:34' ngbvs = coder.nullcopy(zeros(MAXNPNTS,1,'int32')); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:35' vtags = false(nv, 1); */
        nverts = vtags->size[0];
        vtags->size[0] = nv;
        emxEnsureCapacity((emxArray__common *)vtags, nverts, (int32_T)sizeof
                          (boolean_T));
        minpntsv = nv - 1;
        for (nverts = 0; nverts <= minpntsv; nverts++) {
          vtags->data[nverts] = FALSE;
        }

        /* 'polyfit_lhfgrad_surf_cleanmesh:36' ftags = false(size(tris,1), 1); */
        nverts = ftags->size[0];
        ftags->size[0] = tris->size[0];
        emxEnsureCapacity((emxArray__common *)ftags, nverts, (int32_T)sizeof
                          (boolean_T));
        minpntsv = tris->size[0] - 1;
        for (nverts = 0; nverts <= minpntsv; nverts++) {
          ftags->data[nverts] = FALSE;
        }

        /* 'polyfit_lhfgrad_surf_cleanmesh:38' noprdir = nargin<=10 || ~size(prdirs,1); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:40' for ii=1:nv_clean */
        for (ii = 1; ii <= nv_clean; ii++) {
          /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:42' if size(degs,1)>1 && degs(ii)>1 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:43' ringv = ring; */
          ringv = ring;

          /* 'polyfit_lhfgrad_surf_cleanmesh:45' if size(degs,1)>1 && degs(ii)==0 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:48' else */
          /* 'polyfit_lhfgrad_surf_cleanmesh:49' deg_in=degree; */
          /* 'polyfit_lhfgrad_surf_cleanmesh:49' minpntsv =  minpnts; */
          /* 'polyfit_lhfgrad_surf_cleanmesh:52' while (1) */
          do {
            exitg4 = 0U;

            /*  Collect neighbor vertices */
            /* 'polyfit_lhfgrad_surf_cleanmesh:54' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpntsv, ... */
            /* 'polyfit_lhfgrad_surf_cleanmesh:55'             tris, opphes, v2he, ngbvs, vtags, ftags); */
            nverts = b_obtain_nring_surf(ii, ringv, (int32_T)iv15[degree - 1],
              tris, opphes, v2he, ngbvs, vtags, ftags);

            /* 'polyfit_lhfgrad_surf_cleanmesh:57' if noprdir */
            /* 'polyfit_lhfgrad_surf_cleanmesh:58' [deg, prcurvs] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
            polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, degree, &deg,
              prcurvs);

            /* 'polyfit_lhfgrad_surf_cleanmesh:65' if size(curs,1) */
            if (curs->size[0] != 0) {
              /* 'polyfit_lhfgrad_surf_cleanmesh:65' curs(ii,1:2) = prcurvs'; */
              for (nverts = 0; nverts < 2; nverts++) {
                curs->data[(ii + curs->size[0] * nverts) - 1] = prcurvs[nverts];
              }
            }

            /*  Enlarge the neighborhood if necessary */
            /* 'polyfit_lhfgrad_surf_cleanmesh:68' if deg < deg_in && ringv<ring+ring */
            if ((deg < degree) && (ringv < ring + ring)) {
              /* 'polyfit_lhfgrad_surf_cleanmesh:69' ringv=ringv+0.5; */
              ringv += 0.5;
            } else {
              exitg4 = 1U;
            }
          } while (exitg4 == 0U);

          /* 'polyfit_lhfgrad_surf_cleanmesh:70' else */
        }
      } else {
        if (b_min(degs) <= 1) {
          /* 'polyfit_lhf_surf_cleanmesh:114' elseif min(degs,[],1)<=1 */
          /* 'polyfit_lhf_surf_cleanmesh:115' curs = polyfit_lhfgrad_surf_cleanmesh(nv_clean, xs, nrms, tris, opphes, ... */
          /* 'polyfit_lhf_surf_cleanmesh:116'             v2he, degs, degree, ring, curs); */
          /* POLYFIT_LHFGRAD_SURF_CLEANMESH Compute polynomial fitting of gradients with adaptive */
          /* reduced QR factorization. */
          /*  [CURS,PRDIRS] = POLYFIT_LHFGRAD_SURF(XS,NRMS,TRIS,OPPHES,V2HE,DEGS, ... */
          /*  DEGREE,RING,CURS,PRDIRS) Computes polynomial fitting of gradients with  */
          /*  adaptive reduced QR factorization using the following input and output */
          /*  arguments. */
          /*  Input:  XS:       nv*3 Coordinates of points */
          /*          NRMS:     Normals to be fit */
          /*          TRIS:     matrix of size mx3 storing element connectivity */
          /*          OPPTR:    matrix of size mx3 storing opposite vertices */
          /*          DEGREE:   Degree of polynomials */
          /*          RECUR:    Whether or not to use iterative fitting */
          /*          STRIP:    Whether or not to enforce fitting to pass a given point. */
          /*   */
          /*  Output: CURS:     Principal curvatures (nx2); */
          /*          PRDIRS:   Principal directions crt maximum curvature (nx3) */
          /*  */
          /*  See also POLYFIT_LHFGRAD_SURF_POINT */
          /* 'polyfit_lhfgrad_surf_cleanmesh:21' MAXNPNTS=int32(128); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:22' assert(isa(ring,'double')); */
          /*  ring is double, as we allow half rings. */
          /* 'polyfit_lhfgrad_surf_cleanmesh:24' if degree<=6 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:25' pntsneeded = int32([5 9 15 23 32 42]); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:26' minpnts = pntsneeded(degree); */
          /*  Compute fitting at all vertices */
          /* 'polyfit_lhfgrad_surf_cleanmesh:32' nv = int32(size(xs, 1)); */
          nv = xs->size[0];

          /* 'polyfit_lhfgrad_surf_cleanmesh:34' ngbvs = coder.nullcopy(zeros(MAXNPNTS,1,'int32')); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:35' vtags = false(nv, 1); */
          nverts = vtags->size[0];
          vtags->size[0] = nv;
          emxEnsureCapacity((emxArray__common *)vtags, nverts, (int32_T)sizeof
                            (boolean_T));
          minpntsv = nv - 1;
          for (nverts = 0; nverts <= minpntsv; nverts++) {
            vtags->data[nverts] = FALSE;
          }

          /* 'polyfit_lhfgrad_surf_cleanmesh:36' ftags = false(size(tris,1), 1); */
          nverts = ftags->size[0];
          ftags->size[0] = tris->size[0];
          emxEnsureCapacity((emxArray__common *)ftags, nverts, (int32_T)sizeof
                            (boolean_T));
          minpntsv = tris->size[0] - 1;
          for (nverts = 0; nverts <= minpntsv; nverts++) {
            ftags->data[nverts] = FALSE;
          }

          /* 'polyfit_lhfgrad_surf_cleanmesh:38' noprdir = nargin<=10 || ~size(prdirs,1); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:40' for ii=1:nv_clean */
          for (ii = 1; ii <= nv_clean; ii++) {
            /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
            /* 'polyfit_lhfgrad_surf_cleanmesh:42' if size(degs,1)>1 && degs(ii)>1 */
            if ((degs->size[0] > 1) && (degs->data[ii - 1] > 1)) {
            } else {
              /* 'polyfit_lhfgrad_surf_cleanmesh:43' ringv = ring; */
              ringv = ring;

              /* 'polyfit_lhfgrad_surf_cleanmesh:45' if size(degs,1)>1 && degs(ii)==0 */
              if ((degs->size[0] > 1) && (degs->data[ii - 1] == 0)) {
                /*  Use one-ring if degree is 0 */
                /* 'polyfit_lhfgrad_surf_cleanmesh:47' deg_in=int32(0); */
                nv = 0;

                /* 'polyfit_lhfgrad_surf_cleanmesh:47' ringv = 1; */
                ringv = 1.0;

                /* 'polyfit_lhfgrad_surf_cleanmesh:47' minpntsv = int32(0); */
                minpntsv = 0;
              } else {
                /* 'polyfit_lhfgrad_surf_cleanmesh:48' else */
                /* 'polyfit_lhfgrad_surf_cleanmesh:49' deg_in=degree; */
                nv = degree;

                /* 'polyfit_lhfgrad_surf_cleanmesh:49' minpntsv =  minpnts; */
                minpntsv = (int32_T)iv15[degree - 1];
              }

              /* 'polyfit_lhfgrad_surf_cleanmesh:52' while (1) */
              do {
                exitg3 = 0U;

                /*  Collect neighbor vertices */
                /* 'polyfit_lhfgrad_surf_cleanmesh:54' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpntsv, ... */
                /* 'polyfit_lhfgrad_surf_cleanmesh:55'             tris, opphes, v2he, ngbvs, vtags, ftags); */
                nverts = b_obtain_nring_surf(ii, ringv, minpntsv, tris, opphes,
                  v2he, ngbvs, vtags, ftags);

                /* 'polyfit_lhfgrad_surf_cleanmesh:57' if noprdir */
                /* 'polyfit_lhfgrad_surf_cleanmesh:58' [deg, prcurvs] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
                polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, nv, &deg,
                  prcurvs);

                /* 'polyfit_lhfgrad_surf_cleanmesh:65' if size(curs,1) */
                if (curs->size[0] != 0) {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:65' curs(ii,1:2) = prcurvs'; */
                  for (nverts = 0; nverts < 2; nverts++) {
                    curs->data[(ii + curs->size[0] * nverts) - 1] =
                      prcurvs[nverts];
                  }
                }

                /*  Enlarge the neighborhood if necessary */
                /* 'polyfit_lhfgrad_surf_cleanmesh:68' if deg < deg_in && ringv<ring+ring */
                if ((deg < nv) && (ringv < ring + ring)) {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:69' ringv=ringv+0.5; */
                  ringv += 0.5;
                } else {
                  exitg3 = 1U;
                }
              } while (exitg3 == 0U);

              /* 'polyfit_lhfgrad_surf_cleanmesh:70' else */
            }
          }
        }
      }
    } else {
      /* 'polyfit_lhf_surf_cleanmesh:118' else */
      /* 'polyfit_lhf_surf_cleanmesh:119' if iterfit */
      if (iterfit) {
        /* 'polyfit_lhf_surf_cleanmesh:120' [curs,prdirs] = polyfit_lhfgrad_surf_cleanmesh(nv_clean, xs, nrms, tris, opphes, ... */
        /* 'polyfit_lhf_surf_cleanmesh:121'             v2he, degree, degree, ring, curs, prdirs); */
        /* POLYFIT_LHFGRAD_SURF_CLEANMESH Compute polynomial fitting of gradients with adaptive */
        /* reduced QR factorization. */
        /*  [CURS,PRDIRS] = POLYFIT_LHFGRAD_SURF(XS,NRMS,TRIS,OPPHES,V2HE,DEGS, ... */
        /*  DEGREE,RING,CURS,PRDIRS) Computes polynomial fitting of gradients with  */
        /*  adaptive reduced QR factorization using the following input and output */
        /*  arguments. */
        /*  Input:  XS:       nv*3 Coordinates of points */
        /*          NRMS:     Normals to be fit */
        /*          TRIS:     matrix of size mx3 storing element connectivity */
        /*          OPPTR:    matrix of size mx3 storing opposite vertices */
        /*          DEGREE:   Degree of polynomials */
        /*          RECUR:    Whether or not to use iterative fitting */
        /*          STRIP:    Whether or not to enforce fitting to pass a given point. */
        /*   */
        /*  Output: CURS:     Principal curvatures (nx2); */
        /*          PRDIRS:   Principal directions crt maximum curvature (nx3) */
        /*  */
        /*  See also POLYFIT_LHFGRAD_SURF_POINT */
        /* 'polyfit_lhfgrad_surf_cleanmesh:21' MAXNPNTS=int32(128); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:22' assert(isa(ring,'double')); */
        /*  ring is double, as we allow half rings. */
        /* 'polyfit_lhfgrad_surf_cleanmesh:24' if degree<=6 */
        /* 'polyfit_lhfgrad_surf_cleanmesh:25' pntsneeded = int32([5 9 15 23 32 42]); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:26' minpnts = pntsneeded(degree); */
        /*  Compute fitting at all vertices */
        /* 'polyfit_lhfgrad_surf_cleanmesh:32' nv = int32(size(xs, 1)); */
        nv = xs->size[0];

        /* 'polyfit_lhfgrad_surf_cleanmesh:34' ngbvs = coder.nullcopy(zeros(MAXNPNTS,1,'int32')); */
        /* 'polyfit_lhfgrad_surf_cleanmesh:35' vtags = false(nv, 1); */
        nverts = vtags->size[0];
        vtags->size[0] = nv;
        emxEnsureCapacity((emxArray__common *)vtags, nverts, (int32_T)sizeof
                          (boolean_T));
        minpntsv = nv - 1;
        for (nverts = 0; nverts <= minpntsv; nverts++) {
          vtags->data[nverts] = FALSE;
        }

        /* 'polyfit_lhfgrad_surf_cleanmesh:36' ftags = false(size(tris,1), 1); */
        nverts = ftags->size[0];
        ftags->size[0] = tris->size[0];
        emxEnsureCapacity((emxArray__common *)ftags, nverts, (int32_T)sizeof
                          (boolean_T));
        minpntsv = tris->size[0] - 1;
        for (nverts = 0; nverts <= minpntsv; nverts++) {
          ftags->data[nverts] = FALSE;
        }

        /* 'polyfit_lhfgrad_surf_cleanmesh:38' noprdir = nargin<=10 || ~size(prdirs,1); */
        if (!(prdirs->size[0] != 0)) {
          b0 = TRUE;
        } else {
          b0 = FALSE;
        }

        /* 'polyfit_lhfgrad_surf_cleanmesh:40' for ii=1:nv_clean */
        for (ii = 1; ii <= nv_clean; ii++) {
          /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:42' if size(degs,1)>1 && degs(ii)>1 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:43' ringv = ring; */
          ringv = ring;

          /* 'polyfit_lhfgrad_surf_cleanmesh:45' if size(degs,1)>1 && degs(ii)==0 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:48' else */
          /* 'polyfit_lhfgrad_surf_cleanmesh:49' deg_in=degree; */
          /* 'polyfit_lhfgrad_surf_cleanmesh:49' minpntsv =  minpnts; */
          /* 'polyfit_lhfgrad_surf_cleanmesh:52' while (1) */
          do {
            exitg2 = 0U;

            /*  Collect neighbor vertices */
            /* 'polyfit_lhfgrad_surf_cleanmesh:54' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpntsv, ... */
            /* 'polyfit_lhfgrad_surf_cleanmesh:55'             tris, opphes, v2he, ngbvs, vtags, ftags); */
            nverts = b_obtain_nring_surf(ii, ringv, (int32_T)iv15[degree - 1],
              tris, opphes, v2he, ngbvs, vtags, ftags);

            /* 'polyfit_lhfgrad_surf_cleanmesh:57' if noprdir */
            if (b0) {
              /* 'polyfit_lhfgrad_surf_cleanmesh:58' [deg, prcurvs] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
              polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, degree,
                &deg, prcurvs);
            } else {
              /* 'polyfit_lhfgrad_surf_cleanmesh:59' elseif nargout==2 */
              /* 'polyfit_lhfgrad_surf_cleanmesh:60' [deg, prcurvs, maxprdir] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
              b_polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, degree,
                &deg, prcurvs, maxprdir);

              /* 'polyfit_lhfgrad_surf_cleanmesh:62' if size(prdirs,1) */
              if (prdirs->size[0] != 0) {
                /* 'polyfit_lhfgrad_surf_cleanmesh:62' prdirs(ii,1:3) = maxprdir'; */
                for (nverts = 0; nverts < 3; nverts++) {
                  prdirs->data[(ii + prdirs->size[0] * nverts) - 1] =
                    maxprdir[nverts];
                }
              }
            }

            /* 'polyfit_lhfgrad_surf_cleanmesh:65' if size(curs,1) */
            if (curs->size[0] != 0) {
              /* 'polyfit_lhfgrad_surf_cleanmesh:65' curs(ii,1:2) = prcurvs'; */
              for (nverts = 0; nverts < 2; nverts++) {
                curs->data[(ii + curs->size[0] * nverts) - 1] = prcurvs[nverts];
              }
            }

            /*  Enlarge the neighborhood if necessary */
            /* 'polyfit_lhfgrad_surf_cleanmesh:68' if deg < deg_in && ringv<ring+ring */
            if ((deg < degree) && (ringv < ring + ring)) {
              /* 'polyfit_lhfgrad_surf_cleanmesh:69' ringv=ringv+0.5; */
              ringv += 0.5;
            } else {
              exitg2 = 1U;
            }
          } while (exitg2 == 0U);

          /* 'polyfit_lhfgrad_surf_cleanmesh:70' else */
        }
      } else {
        if (b_min(degs) <= 1) {
          /* 'polyfit_lhf_surf_cleanmesh:122' elseif min(degs,[],1)<=1 */
          /* 'polyfit_lhf_surf_cleanmesh:123' [curs,prdirs] = polyfit_lhfgrad_surf_cleanmesh(nv_clean, xs, nrms, tris, opphes, ... */
          /* 'polyfit_lhf_surf_cleanmesh:124'             v2he, degs, degree, ring, curs, prdirs); */
          /* POLYFIT_LHFGRAD_SURF_CLEANMESH Compute polynomial fitting of gradients with adaptive */
          /* reduced QR factorization. */
          /*  [CURS,PRDIRS] = POLYFIT_LHFGRAD_SURF(XS,NRMS,TRIS,OPPHES,V2HE,DEGS, ... */
          /*  DEGREE,RING,CURS,PRDIRS) Computes polynomial fitting of gradients with  */
          /*  adaptive reduced QR factorization using the following input and output */
          /*  arguments. */
          /*  Input:  XS:       nv*3 Coordinates of points */
          /*          NRMS:     Normals to be fit */
          /*          TRIS:     matrix of size mx3 storing element connectivity */
          /*          OPPTR:    matrix of size mx3 storing opposite vertices */
          /*          DEGREE:   Degree of polynomials */
          /*          RECUR:    Whether or not to use iterative fitting */
          /*          STRIP:    Whether or not to enforce fitting to pass a given point. */
          /*   */
          /*  Output: CURS:     Principal curvatures (nx2); */
          /*          PRDIRS:   Principal directions crt maximum curvature (nx3) */
          /*  */
          /*  See also POLYFIT_LHFGRAD_SURF_POINT */
          /* 'polyfit_lhfgrad_surf_cleanmesh:21' MAXNPNTS=int32(128); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:22' assert(isa(ring,'double')); */
          /*  ring is double, as we allow half rings. */
          /* 'polyfit_lhfgrad_surf_cleanmesh:24' if degree<=6 */
          /* 'polyfit_lhfgrad_surf_cleanmesh:25' pntsneeded = int32([5 9 15 23 32 42]); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:26' minpnts = pntsneeded(degree); */
          /*  Compute fitting at all vertices */
          /* 'polyfit_lhfgrad_surf_cleanmesh:32' nv = int32(size(xs, 1)); */
          nv = xs->size[0];

          /* 'polyfit_lhfgrad_surf_cleanmesh:34' ngbvs = coder.nullcopy(zeros(MAXNPNTS,1,'int32')); */
          /* 'polyfit_lhfgrad_surf_cleanmesh:35' vtags = false(nv, 1); */
          nverts = vtags->size[0];
          vtags->size[0] = nv;
          emxEnsureCapacity((emxArray__common *)vtags, nverts, (int32_T)sizeof
                            (boolean_T));
          minpntsv = nv - 1;
          for (nverts = 0; nverts <= minpntsv; nverts++) {
            vtags->data[nverts] = FALSE;
          }

          /* 'polyfit_lhfgrad_surf_cleanmesh:36' ftags = false(size(tris,1), 1); */
          nverts = ftags->size[0];
          ftags->size[0] = tris->size[0];
          emxEnsureCapacity((emxArray__common *)ftags, nverts, (int32_T)sizeof
                            (boolean_T));
          minpntsv = tris->size[0] - 1;
          for (nverts = 0; nverts <= minpntsv; nverts++) {
            ftags->data[nverts] = FALSE;
          }

          /* 'polyfit_lhfgrad_surf_cleanmesh:38' noprdir = nargin<=10 || ~size(prdirs,1); */
          if (!(prdirs->size[0] != 0)) {
            b0 = TRUE;
          } else {
            b0 = FALSE;
          }

          /* 'polyfit_lhfgrad_surf_cleanmesh:40' for ii=1:nv_clean */
          for (ii = 1; ii <= nv_clean; ii++) {
            /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
            /* 'polyfit_lhfgrad_surf_cleanmesh:42' if size(degs,1)>1 && degs(ii)>1 */
            if ((degs->size[0] > 1) && (degs->data[ii - 1] > 1)) {
            } else {
              /* 'polyfit_lhfgrad_surf_cleanmesh:43' ringv = ring; */
              ringv = ring;

              /* 'polyfit_lhfgrad_surf_cleanmesh:45' if size(degs,1)>1 && degs(ii)==0 */
              if ((degs->size[0] > 1) && (degs->data[ii - 1] == 0)) {
                /*  Use one-ring if degree is 0 */
                /* 'polyfit_lhfgrad_surf_cleanmesh:47' deg_in=int32(0); */
                nv = 0;

                /* 'polyfit_lhfgrad_surf_cleanmesh:47' ringv = 1; */
                ringv = 1.0;

                /* 'polyfit_lhfgrad_surf_cleanmesh:47' minpntsv = int32(0); */
                minpntsv = 0;
              } else {
                /* 'polyfit_lhfgrad_surf_cleanmesh:48' else */
                /* 'polyfit_lhfgrad_surf_cleanmesh:49' deg_in=degree; */
                nv = degree;

                /* 'polyfit_lhfgrad_surf_cleanmesh:49' minpntsv =  minpnts; */
                minpntsv = (int32_T)iv15[degree - 1];
              }

              /* 'polyfit_lhfgrad_surf_cleanmesh:52' while (1) */
              do {
                exitg1 = 0U;

                /*  Collect neighbor vertices */
                /* 'polyfit_lhfgrad_surf_cleanmesh:54' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpntsv, ... */
                /* 'polyfit_lhfgrad_surf_cleanmesh:55'             tris, opphes, v2he, ngbvs, vtags, ftags); */
                nverts = b_obtain_nring_surf(ii, ringv, minpntsv, tris, opphes,
                  v2he, ngbvs, vtags, ftags);

                /* 'polyfit_lhfgrad_surf_cleanmesh:57' if noprdir */
                if (b0) {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:58' [deg, prcurvs] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
                  polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, nv,
                    &deg, prcurvs);
                } else {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:59' elseif nargout==2 */
                  /* 'polyfit_lhfgrad_surf_cleanmesh:60' [deg, prcurvs, maxprdir] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
                  b_polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, nv,
                    &deg, prcurvs, maxprdir);

                  /* 'polyfit_lhfgrad_surf_cleanmesh:62' if size(prdirs,1) */
                  if (prdirs->size[0] != 0) {
                    /* 'polyfit_lhfgrad_surf_cleanmesh:62' prdirs(ii,1:3) = maxprdir'; */
                    for (nverts = 0; nverts < 3; nverts++) {
                      prdirs->data[(ii + prdirs->size[0] * nverts) - 1] =
                        maxprdir[nverts];
                    }
                  }
                }

                /* 'polyfit_lhfgrad_surf_cleanmesh:65' if size(curs,1) */
                if (curs->size[0] != 0) {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:65' curs(ii,1:2) = prcurvs'; */
                  for (nverts = 0; nverts < 2; nverts++) {
                    curs->data[(ii + curs->size[0] * nverts) - 1] =
                      prcurvs[nverts];
                  }
                }

                /*  Enlarge the neighborhood if necessary */
                /* 'polyfit_lhfgrad_surf_cleanmesh:68' if deg < deg_in && ringv<ring+ring */
                if ((deg < nv) && (ringv < ring + ring)) {
                  /* 'polyfit_lhfgrad_surf_cleanmesh:69' ringv=ringv+0.5; */
                  ringv += 0.5;
                } else {
                  exitg1 = 1U;
                }
              } while (exitg1 == 0U);

              /* 'polyfit_lhfgrad_surf_cleanmesh:70' else */
            }
          }
        }
      }
    }
  }

  emxFree_int32_T(&degs);
  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

/*
 * function [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point...
 *     (v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
 */
static void polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128], int32_T
  nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor, int32_T
  degree, real_T nrm[3], int32_T *deg)
{
  int32_T i;
  int32_T ix;
  real_T absnrm[3];
  static const int8_T iv0[3] = { 0, 1, 0 };

  static const int8_T iv1[3] = { 1, 0, 0 };

  real_T y;
  int32_T b_ix;
  int32_T iy;
  int32_T k;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws_row;
  real_T t2[3];
  real_T cs2[3];
  real_T grad[2];
  real_T nrm_l[3];
  real_T P[9];
  real_T b_y;
  int32_T b_iy;
  real_T c_y;
  int32_T c_iy;
  emxArray_real_T *b_us;
  real_T unusedExpr[3];

  /* POLYFIT_LHF_SURF_POINT Compute normal, principal curvatures, and principal */
  /* direction. */
  /*  [NRM,DEG,PRCURVS,MAXPRDIR] = POLYFIT_LHF_SURF_POINT(V,NGBVS,NVERTS,XS, ... */
  /*  NRMS_COOR, DEGREE, INTERP, GUARDOSC) Computes normal NRM, principal */
  /*  curvatures PRCURVS, and principal direction MAXPRDIR at vertex, given list */
  /*  of neighbor points. */
  /*  */
  /*  See also POLYFIT_LHF_SURF */
  /* 'polyfit_lhf_surf_point:12' ngbvs = int32(ngbvs); */
  /* added */
  /* 'polyfit_lhf_surf_point:13' MAXNPNTS=int32(128); */
  /* 'polyfit_lhf_surf_point:14' assert( isa( v, 'int32') && isa( ngbvs, 'int32')); */
  /* 'polyfit_lhf_surf_point:15' assert( isa( nverts, 'int32') && isa( degree, 'int32')); */
  /* 'polyfit_lhf_surf_point:17' if nargin<8 */
  /* 'polyfit_lhf_surf_point:17' guardosc=false; */
  /* 'polyfit_lhf_surf_point:19' if nverts==0 */
  if (nverts == 0) {
    /* 'polyfit_lhf_surf_point:20' nrm = [0; 0; 0]; */
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:20' deg = int32(0); */
    *deg = 0;

    /* 'polyfit_lhf_surf_point:21' prcurvs = [0;0]; */
    /* 'polyfit_lhf_surf_point:21' maxprdir = [0;0;0]; */
  } else {
    if (nverts >= 128) {
      /* 'polyfit_lhf_surf_point:23' elseif nverts>=MAXNPNTS */
      /* 'polyfit_lhf_surf_point:24' nverts = MAXNPNTS-1; */
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    /* 'polyfit_lhf_surf_point:28' nrm = nrms_coor(v,1:3)'; */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    /* 'polyfit_lhf_surf_point:29' absnrm = abs(nrm); */
    b_abs(nrm, absnrm);

    /* 'polyfit_lhf_surf_point:31' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      /* 'polyfit_lhf_surf_point:32' t1 = [0; 1; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv0[i];
      }
    } else {
      /* 'polyfit_lhf_surf_point:33' else */
      /* 'polyfit_lhf_surf_point:34' t1 = [1; 0; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv1[i];
      }
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 - t1' * nrm * nrm; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * nrm[iy];
      b_ix++;
      iy++;
    }

    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] -= y * nrm[ix];
    }

    /* 'polyfit_lhf_surf_point:37' t1 = t1 / sqrt(t1'*t1); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += absnrm[b_ix] * absnrm[iy];
      b_ix++;
      iy++;
    }

    y = sqrt(y);
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] /= y;
    }

    emxInit_real_T(&us, 2);
    b_emxInit_real_T(&bs, 1);
    b_emxInit_real_T(&ws_row, 1);

    /* 'polyfit_lhf_surf_point:38' t2 = cross_col( nrm, t1); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    /* 'polyfit_lhf_surf_point:41' us = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:42' bs = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:43' ws_row = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhf_surf_point:45' us(1,:)=0; */
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    /* 'polyfit_lhf_surf_point:45' ws_row(1)=1; */
    ws_row->data[0] = 1.0;

    /* 'polyfit_lhf_surf_point:46' for ii=1:nverts */
    for (i = 0; i + 1 <= nverts; i++) {
      /* 'polyfit_lhf_surf_point:47' u = xs(ngbvs(ii),1:3)-xs(v,1:3); */
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      /* 'polyfit_lhf_surf_point:49' us(ii+1-int32(interp),1) = u*t1; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;

      /* 'polyfit_lhf_surf_point:50' us(ii+1-int32(interp),2) = u*t2; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;

      /* 'polyfit_lhf_surf_point:51' bs(ii+1-int32(interp)) = u*nrm; */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * nrm[iy];
        b_ix++;
        iy++;
      }

      bs->data[i] = y;

      /*  Compute normal-based weights */
      /* 'polyfit_lhf_surf_point:54' ws_row(ii+1-int32(interp)) = max(0, nrms_coor(ngbvs(ii),:)*nrm); */
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += nrms_coor->data[(ngbvs[i] + nrms_coor->size[0] * b_ix) - 1] *
          nrm[iy];
        b_ix++;
        iy++;
      }

      y = 0.0 >= y ? 0.0 : y;
      ws_row->data[i] = y;
    }

    /* 'polyfit_lhf_surf_point:57' if degree==0 */
    if (degree == 0) {
      /*  Use linear fitting without weight */
      /* 'polyfit_lhf_surf_point:59' ws_row(:) = 1; */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      /* 'polyfit_lhf_surf_point:59' degree=int32(1); */
      degree = 1;
    }

    /*  Compute the coefficients */
    /* 'polyfit_lhf_surf_point:63' [bs, deg] = eval_vander_bivar( us, bs, degree, ws_row, interp, guardosc); */
    *deg = degree;
    eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    /* 'polyfit_lhf_surf_point:66' if deg<=1 */
    /* 'polyfit_lhf_surf_point:67' coder.varsize('cs', [6,1],[1,0]); */
    /* 'polyfit_lhf_surf_point:68' cs = bs(2-int32(interp):n); */
    /* 'polyfit_lhf_surf_point:70' grad = [cs(1); cs(2)]; */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];

    /* 'polyfit_lhf_surf_point:71' nrm_l = [-grad; 1]/sqrt(1+grad'*grad); */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    emxFree_real_T(&bs);
    for (k = 0; k < 2; k++) {
      y += grad[b_ix] * grad[iy];
      b_ix++;
      iy++;
    }

    y = sqrt(1.0 + y);
    for (i = 0; i < 2; i++) {
      nrm_l[i] = -grad[i] / y;
    }

    nrm_l[2] = 1.0 / y;

    /* 'polyfit_lhf_surf_point:73' P = [t1, t2, nrm]; */
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
    /* 'polyfit_lhf_surf_point:75' nrm = [P(1,:) * nrm_l; P(2,:) * nrm_l; P(3,:) * nrm_l]; */
    y = 0.0;
    b_ix = 0;
    iy = 0;
    b_y = 0.0;
    i = 0;
    b_iy = 0;
    c_y = 0.0;
    ix = 0;
    c_iy = 0;
    for (k = 0; k < 3; k++) {
      y += P[3 * b_ix] * nrm_l[iy];
      b_ix++;
      iy++;
      b_y += P[1 + 3 * i] * nrm_l[b_iy];
      i++;
      b_iy++;
      c_y += P[2 + 3 * ix] * nrm_l[c_iy];
      ix++;
      c_iy++;
    }

    nrm[0] = y;
    nrm[1] = b_y;
    nrm[2] = c_y;

    /* 'polyfit_lhf_surf_point:77' if deg>1 */
    if ((*deg > 1) || (!(nverts >= 2))) {
      /* 'polyfit_lhf_surf_point:78' H = [2*cs(3) cs(4); cs(4) 2*cs(5)]; */
      /* 'polyfit_lhf_surf_point:88' else */
      /* 'polyfit_lhf_surf_point:89' H = coder.nullcopy(zeros(2,2)); */
    } else {
      /* 'polyfit_lhf_surf_point:79' elseif deg<=1 && nverts>=2 */
      /* 'polyfit_lhf_surf_point:80' if deg==0 && nverts>=2 */
      if (*deg == 0) {
        emxInit_real_T(&b_us, 2);

        /* 'polyfit_lhf_surf_point:81' us = us(1:3-int32(interp),:); */
        ix = b_us->size[0] * b_us->size[1];
        b_us->size[0] = 2;
        b_us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)b_us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          for (b_iy = 0; b_iy < 2; b_iy++) {
            b_us->data[b_iy + b_us->size[0] * ix] = us->data[b_iy + us->size[0] *
              ix];
          }
        }

        ix = us->size[0] * us->size[1];
        us->size[0] = b_us->size[0];
        us->size[1] = 2;
        emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
        for (ix = 0; ix < 2; ix++) {
          i = b_us->size[0] - 1;
          for (b_iy = 0; b_iy <= i; b_iy++) {
            us->data[b_iy + us->size[0] * ix] = b_us->data[b_iy + b_us->size[0] *
              ix];
          }
        }

        emxFree_real_T(&b_us);

        /* 'polyfit_lhf_surf_point:82' ws_row(1:3-int32(interp)) = 1; */
        for (ix = 0; ix < 2; ix++) {
          ws_row->data[ix] = 1.0;
        }
      }

      /*  Try to compute curvatures from normals */
      /* 'polyfit_lhf_surf_point:86' cs2 = linfit_lhf_grad_surf_point( ngbvs, us, t1, t2, nrms_coor, ws_row, interp); */
      linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row,
        unusedExpr);

      /* 'polyfit_lhf_surf_point:87' H = [cs2(1) cs2(2); cs2(2) cs2(3)]; */
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&us);

    /* 'polyfit_lhf_surf_point:92' if deg>=1 */
  }
}

/*
 * function [deg, prcurvs, maxprdir] = polyfit_lhfgrad_surf_point...
 *     ( v, ngbvs, nverts, xs, nrms, degree, interp, guardosc)
 */
static void polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2])
{
  int32_T i;
  int32_T ix;
  real_T nrm[3];
  real_T absnrm[3];
  static const int8_T iv8[3] = { 0, 1, 0 };

  static const int8_T iv9[3] = { 1, 0, 0 };

  real_T y;
  int32_T iy;
  real_T h12;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws_row;
  real_T t2[3];
  int32_T ii;
  real_T u[3];
  real_T grad[2];
  real_T H[4];
  real_T grad_sqnorm;
  real_T ell;
  real_T c;
  real_T s;
  real_T b[2];
  real_T a[2];

  /* POLYFIT_LHFGRAD_SURF_POINT Compute principal curvatures and principal  */
  /* direction. */
  /*  [DEG,PRCURVS,MAXPRDIR] = POLYFIT_LHFGRAD_SURF_POINT(V,NGBVS,NVERTS, ... */
  /*  XS,NRMS,DEGREE,INTERP,GUARDOSC) Computes principal curvatures and */
  /*  principal direction at vertex, using given points XS and vertex normals NRMS. */
  /*  */
  /*  See also POLYFIT_LHFGRAD_SURF_POINT */
  /* 'polyfit_lhfgrad_surf_point:11' MAXNPNTS=int32(128); */
  /* 'polyfit_lhfgrad_surf_point:13' if nargin<8 */
  /* 'polyfit_lhfgrad_surf_point:13' guardosc=false; */
  /* 'polyfit_lhfgrad_surf_point:15' if nverts==0 */
  if (nverts == 0) {
    /* 'polyfit_lhfgrad_surf_point:16' deg = int32(0); */
    *deg = 0;

    /* 'polyfit_lhfgrad_surf_point:16' prcurvs = [0;0]; */
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:16' maxprdir = [0;0;0]; */
  } else {
    if (nverts >= 128) {
      /* 'polyfit_lhfgrad_surf_point:18' elseif nverts>=MAXNPNTS */
      /* 'polyfit_lhfgrad_surf_point:19' nverts = MAXNPNTS-1+int32(interp); */
      nverts = 127;
    }

    /*  First, compute the rotation matrix */
    /* 'polyfit_lhfgrad_surf_point:23' nrm = nrms(v,1:3)'; */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms->data[(v + nrms->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    /* 'polyfit_lhfgrad_surf_point:24' absnrm = abs(nrm); */
    for (i = 0; i < 3; i++) {
      absnrm[i] = fabs(nrm[i]);
    }

    /* 'polyfit_lhfgrad_surf_point:26' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      /* 'polyfit_lhfgrad_surf_point:27' t1 = [0; 1; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv8[i];
      }
    } else {
      /* 'polyfit_lhfgrad_surf_point:28' else */
      /* 'polyfit_lhfgrad_surf_point:29' t1 = [1; 0; 0]; */
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv9[i];
      }
    }

    /* 'polyfit_lhfgrad_surf_point:32' t1 = t1 - t1' * nrm * nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (i = 0; i < 3; i++) {
      y += absnrm[ix] * nrm[iy];
      ix++;
      iy++;
    }

    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] -= y * nrm[ix];
    }

    /* 'polyfit_lhfgrad_surf_point:32' t1 = t1 / sqrt(t1'*t1); */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (i = 0; i < 3; i++) {
      y += absnrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    h12 = sqrt(y);
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] /= h12;
    }

    emxInit_real_T(&us, 2);
    emxInit_real_T(&bs, 2);
    b_emxInit_real_T(&ws_row, 1);

    /* 'polyfit_lhfgrad_surf_point:33' t2 = cross_col( nrm, t1); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Evaluate local coordinate system and weights */
    /* 'polyfit_lhfgrad_surf_point:36' us = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts + 1;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:37' bs = coder.nullcopy(zeros( nverts+1-int32(interp),2)); */
    ix = bs->size[0] * bs->size[1];
    bs->size[0] = nverts + 1;
    bs->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:38' ws_row = coder.nullcopy(zeros( nverts+1-int32(interp),1)); */
    ix = ws_row->size[0];
    ws_row->size[0] = nverts + 1;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));

    /* 'polyfit_lhfgrad_surf_point:40' if ~interp */
    /* 'polyfit_lhfgrad_surf_point:41' us(1,:)=0; */
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:41' bs(1,:)=0; */
    for (ix = 0; ix < 2; ix++) {
      bs->data[bs->size[0] * ix] = 0.0;
    }

    /* 'polyfit_lhfgrad_surf_point:41' ws_row(1) = 1; */
    ws_row->data[0] = 1.0;

    /* 'polyfit_lhfgrad_surf_point:44' for ii=1:nverts */
    for (ii = 1; ii <= nverts; ii++) {
      /* 'polyfit_lhfgrad_surf_point:45' u = xs(ngbvs(ii),1:3)-xs(v,1:3); */
      for (ix = 0; ix < 3; ix++) {
        u[ix] = xs->data[(ngbvs[ii - 1] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      /* 'polyfit_lhfgrad_surf_point:47' us(ii+1-int32(interp),1) = u*t1; */
      y = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        y += u[ix] * absnrm[iy];
        ix++;
        iy++;
      }

      us->data[ii] = y;

      /* 'polyfit_lhfgrad_surf_point:48' us(ii+1-int32(interp),2) = u*t2; */
      y = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        y += u[ix] * t2[iy];
        ix++;
        iy++;
      }

      us->data[ii + us->size[0]] = y;

      /* 'polyfit_lhfgrad_surf_point:50' nrm_ii = nrms(ngbvs(ii),1:3); */
      /* 'polyfit_lhfgrad_surf_point:51' w = nrm_ii*nrm; */
      h12 = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        h12 += nrms->data[(ngbvs[ii - 1] + nrms->size[0] * ix) - 1] * nrm[iy];
        ix++;
        iy++;
      }

      /* 'polyfit_lhfgrad_surf_point:53' if w>0 */
      if (h12 > 0.0) {
        /* 'polyfit_lhfgrad_surf_point:54' bs(ii+1-int32(interp),1) = -(nrm_ii*t1)/w; */
        y = 0.0;
        ix = 0;
        iy = 0;
        for (i = 0; i < 3; i++) {
          y += nrms->data[(ngbvs[ii - 1] + nrms->size[0] * ix) - 1] * absnrm[iy];
          ix++;
          iy++;
        }

        bs->data[ii] = -y / h12;

        /* 'polyfit_lhfgrad_surf_point:55' bs(ii+1-int32(interp),2) = -(nrm_ii*t2)/w; */
        y = 0.0;
        ix = 0;
        iy = 0;
        for (i = 0; i < 3; i++) {
          y += nrms->data[(ngbvs[ii - 1] + nrms->size[0] * ix) - 1] * t2[iy];
          ix++;
          iy++;
        }

        bs->data[ii + bs->size[0]] = -y / h12;
      }

      /* 'polyfit_lhfgrad_surf_point:57' ws_row(ii+1-int32(interp)) = max(0,w); */
      y = 0.0 >= h12 ? 0.0 : h12;
      ws_row->data[ii] = y;
    }

    /* 'polyfit_lhfgrad_surf_point:60' if degree==0 */
    if (degree == 0) {
      /*  Use linear fitting without weight */
      /* 'polyfit_lhfgrad_surf_point:62' ws_row(:) = 1; */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      /* 'polyfit_lhfgrad_surf_point:62' degree=int32(1); */
      degree = 1;
    }

    /*  Compute the coefficients and store them */
    /* 'polyfit_lhfgrad_surf_point:66' [bs, deg] = eval_vander_bivar( us, bs, degree, ws_row, interp, guardosc); */
    *deg = degree;
    c_eval_vander_bivar(us, bs, deg, ws_row);

    /* 'polyfit_lhfgrad_surf_point:68' if interp */
    /* 'polyfit_lhfgrad_surf_point:73' else */
    /*  Convert coefficients into normals and curvatures */
    /* 'polyfit_lhfgrad_surf_point:75' grad = [bs(1,1) bs(1,2)]; */
    grad[0] = bs->data[0];
    grad[1] = bs->data[bs->size[0]];

    /* 'polyfit_lhfgrad_surf_point:76' h12 = 0.5*(bs(3,1)+bs(2,2)); */
    h12 = bs->data[2] + bs->data[1 + bs->size[0]];
    h12 *= 0.5;

    /* 'polyfit_lhfgrad_surf_point:77' H = [bs(2,1) h12; h12 bs(3,2)]; */
    H[0] = bs->data[1];
    H[2] = h12;
    H[1] = h12;
    H[3] = bs->data[2 + bs->size[0]];

    /* 'polyfit_lhfgrad_surf_point:80' if nargout<=2 */
    /* 'polyfit_lhfgrad_surf_point:81' prcurvs = eval_curvature_lhf_surf(grad, H); */
    /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
    /* and pseudo-inverse. */
    /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
    /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
    /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
    /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
    /*  height function in 2x2 H with a local coordinate frame. */
    /*  */
    /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
    /* 'eval_curvature_lhf_surf:12' grad_sqnorm = grad(1)^2+grad(2)^2; */
    h12 = grad[0];
    y = pow(h12, 2.0);
    h12 = grad[1];
    h12 = pow(h12, 2.0);
    grad_sqnorm = y + h12;

    /* 'eval_curvature_lhf_surf:13' grad_norm = sqrt(grad_sqnorm); */
    h12 = sqrt(grad_sqnorm);

    /*  Compute key parameters */
    /* 'eval_curvature_lhf_surf:16' ell = sqrt(1+grad_sqnorm); */
    ell = sqrt(1.0 + grad_sqnorm);

    /* 'eval_curvature_lhf_surf:17' ell2=1+grad_sqnorm; */
    /* 'eval_curvature_lhf_surf:17' ell3 = ell*(1+grad_sqnorm); */
    /* 'eval_curvature_lhf_surf:18' if grad_norm==0 */
    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    if (h12 == 0.0) {
      /* 'eval_curvature_lhf_surf:19' c = 1; */
      c = 1.0;

      /* 'eval_curvature_lhf_surf:19' s=0; */
      s = 0.0;
    } else {
      /* 'eval_curvature_lhf_surf:20' else */
      /* 'eval_curvature_lhf_surf:21' c = grad(1)/grad_norm; */
      c = grad[0] / h12;

      /* 'eval_curvature_lhf_surf:21' s = grad(2)/grad_norm; */
      s = grad[1] / h12;
    }

    /*  Compute mean curvature and Gaussian curvature */
    /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
    /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
    /*  Solve quadratic equation to compute principal curvatures */
    /* 'eval_curvature_lhf_surf:29' v = [c*H(1,1)+s*H(1,2) c*H(1,2)+s*H(2,2)]; */
    grad[0] = c * H[0] + s * H[2];
    grad[1] = c * H[2] + s * H[3];

    /* 'eval_curvature_lhf_surf:30' W1 = [v*[c; s]/ell3, v*[-s; c]/ell2]; */
    b[0] = c;
    b[1] = s;
    y = 0.0;
    ix = 0;
    iy = 0;
    for (i = 0; i < 2; i++) {
      y += grad[ix] * b[iy];
      ix++;
      iy++;
    }

    b[0] = -s;
    b[1] = c;
    h12 = 0.0;
    ix = 0;
    iy = 0;
    for (i = 0; i < 2; i++) {
      h12 += grad[ix] * b[iy];
      ix++;
      iy++;
    }

    grad[0] = y / (ell * (1.0 + grad_sqnorm));
    grad[1] = h12 / (1.0 + grad_sqnorm);

    /* 'eval_curvature_lhf_surf:31' W = [W1; W1(2) [c*H(1,2)-s*H(1,1), c*H(2,2)-s*H(1,2)]*[-s; c]/ell]; */
    a[0] = c * H[2] - s * H[0];
    a[1] = c * H[3] - s * H[2];
    b[0] = -s;
    b[1] = c;
    y = 0.0;
    ix = 0;
    iy = 0;
    for (i = 0; i < 2; i++) {
      y += a[ix] * b[iy];
      ix++;
      iy++;
      H[i << 1] = grad[i];
    }

    H[1] = grad[1];
    H[3] = y / ell;

    /*  Lambda = eig(W); */
    /* 'eval_curvature_lhf_surf:34' kH2 = W(1,1)+W(2,2); */
    h12 = H[0] + H[3];

    /* 'eval_curvature_lhf_surf:35' tmp = sqrt((W(1,1)-W(2,2))*(W(1,1)-W(2,2))+4*W(1,2)*W(1,2)); */
    s = sqrt((H[0] - H[3]) * (H[0] - H[3]) + 4.0 * H[2] * H[2]);

    /* 'eval_curvature_lhf_surf:36' if kH2>0 */
    if (h12 > 0.0) {
      /* 'eval_curvature_lhf_surf:37' curvs = 0.5*[kH2+tmp; kH2-tmp]; */
      prcurvs[0] = 0.5 * (h12 + s);
      prcurvs[1] = 0.5 * (h12 - s);
    } else {
      /* 'eval_curvature_lhf_surf:38' else */
      /* 'eval_curvature_lhf_surf:39' curvs = 0.5*[kH2-tmp; kH2+tmp]; */
      prcurvs[0] = 0.5 * (h12 - s);
      prcurvs[1] = 0.5 * (h12 + s);
    }

    /* 'eval_curvature_lhf_surf:42' if nargout > 1 */
  }
}

/*
 * function [A, D, rnk] = qr_safeguarded(A, ncols, D, tol)
 */
static int32_T qr_safeguarded(emxArray_real_T *A, int32_T ncols, emxArray_real_T
  *D)
{
  int32_T rnk;
  emxArray_real_T *v;
  int32_T nrows;
  int32_T jj;
  int32_T k;
  boolean_T exitg1;
  int32_T nv;
  real_T t2;
  real_T t;
  int32_T ii;
  b_emxInit_real_T(&v, 1);

  /*  Compute Householder QR factorization with safeguards. */
  /*  It compares the diagonal entries with the given tolerance to */
  /*  determine whether the matrix is nearly singular. It is */
  /*  specialized for performing polynomial fittings. */
  /*  */
  /*  It saves Householder reflector vectors into lower triangular part A. */
  /*  Save diagonal part of R into D, and upper triangular part (excluding */
  /*  diagonal) of R into upper triangular part of A. */
  /* 'qr_safeguarded:11' if nargin<2 */
  /* 'qr_safeguarded:12' if nargin<3 */
  /* 'qr_safeguarded:13' if nargin<4 */
  /* 'qr_safeguarded:13' tol=1.e-8; */
  /* 'qr_safeguarded:15' rnk = ncols; */
  rnk = ncols;

  /* 'qr_safeguarded:16' nrows = int32(size(A,1)); */
  nrows = A->size[0];

  /* 'qr_safeguarded:17' v = coder.nullcopy(zeros(nrows,1)); */
  jj = v->size[0];
  v->size[0] = nrows;
  emxEnsureCapacity((emxArray__common *)v, jj, (int32_T)sizeof(real_T));

  /* 'qr_safeguarded:19' for k=1:ncols */
  k = 0;
  exitg1 = 0U;
  while ((exitg1 == 0U) && (k + 1 <= ncols)) {
    /* 'qr_safeguarded:20' nv = nrows-k+1; */
    nv = nrows - k;

    /* 'qr_safeguarded:21' for jj=1:nv */
    for (jj = 0; jj + 1 <= nv; jj++) {
      /* 'qr_safeguarded:21' v(jj) = A(jj+k-1,k); */
      v->data[jj] = A->data[(jj + k) + A->size[0] * k];
    }

    /*  We don't need to worry about overflow, since A has been rescaled. */
    /* 'qr_safeguarded:24' t2 = 0; */
    t2 = 0.0;

    /* 'qr_safeguarded:24' for jj=1:nv */
    for (jj = 0; jj + 1 <= nv; jj++) {
      /* 'qr_safeguarded:24' t2 = t2+v(jj)*v(jj); */
      t2 += v->data[jj] * v->data[jj];
    }

    /* 'qr_safeguarded:25' t = sqrt(t2); */
    t = sqrt(t2);

    /* 'qr_safeguarded:27' if v(1)>=0 */
    if (v->data[0] >= 0.0) {
      /* 'qr_safeguarded:28' vnrm2 = sqrt(2*(t2 + v(1)*t)); */
      t2 = sqrt(2.0 * (t2 + v->data[0] * t));

      /* 'qr_safeguarded:29' v(1) = v(1) + t; */
      v->data[0] += t;
    } else {
      /* 'qr_safeguarded:30' else */
      /* 'qr_safeguarded:31' vnrm2 = sqrt(2*(t2 - v(1)*t)); */
      t2 = sqrt(2.0 * (t2 - v->data[0] * t));

      /* 'qr_safeguarded:32' v(1) = v(1) - t; */
      v->data[0] -= t;
    }

    /* 'qr_safeguarded:35' if vnrm2>0 */
    if (t2 > 0.0) {
      /* 'qr_safeguarded:35' for jj=1:nv */
      for (jj = 0; jj + 1 <= nv; jj++) {
        /* 'qr_safeguarded:35' v(jj) = v(jj) / vnrm2; */
        v->data[jj] /= t2;
      }
    }

    /*  Optimized version for */
    /*  A(k:npnts,k:ncols) = A(k:npnts,k:ncols) - 2*v*(v'*A(k:npnts,k:ncols)); */
    /* 'qr_safeguarded:39' for jj=k:ncols */
    for (jj = k; jj + 1 <= ncols; jj++) {
      /* 'qr_safeguarded:40' t2 = 0; */
      t2 = 0.0;

      /* 'qr_safeguarded:40' for ii=1:nv */
      for (ii = 0; ii + 1 <= nv; ii++) {
        /* 'qr_safeguarded:40' t2 = t2+v(ii)*A(ii+k-1,jj); */
        t2 += v->data[ii] * A->data[(ii + k) + A->size[0] * jj];
      }

      /* 'qr_safeguarded:41' t2 = t2+t2; */
      t2 += t2;

      /* 'qr_safeguarded:42' for ii=1:nv */
      for (ii = 0; ii + 1 <= nv; ii++) {
        /* 'qr_safeguarded:42' A(ii+k-1,jj) = A(ii+k-1,jj) - t2 * v(ii); */
        A->data[(ii + k) + A->size[0] * jj] -= t2 * v->data[ii];
      }
    }

    /* 'qr_safeguarded:45' D(k) = A(k,k); */
    D->data[k] = A->data[k + A->size[0] * k];

    /* 'qr_safeguarded:46' for i=1:nv */
    for (jj = 0; jj + 1 <= nv; jj++) {
      /* 'qr_safeguarded:46' A(i+k-1,k) = v(i); */
      A->data[(jj + k) + A->size[0] * k] = v->data[jj];
    }

    /*  Estimate rank of matrix */
    /* 'qr_safeguarded:49' if abs(D(k))<tol && rnk==ncols */
    if (fabs(D->data[k]) < 1.0E-8) {
      /* 'qr_safeguarded:50' rnk = k-1; */
      rnk = k;
      exitg1 = 1U;
    } else {
      k++;
    }
  }

  emxFree_real_T(&v);
  return rnk;
}

/*
 * function [V, ts] = rescale_matrix(V, ncols, ts)
 */
static void rescale_matrix(emxArray_real_T *V, int32_T ncols, emxArray_real_T
  *ts)
{
  int32_T ii;
  emxArray_real_T *b_V;
  int32_T kk;
  int32_T loop_ub;

  /* % Rescale the columns of a matrix to reduce condition number */
  /* 'rescale_matrix:4' if nargin<2 */
  /* 'rescale_matrix:7' if nargin<3 */
  /* 'rescale_matrix:9' else */
  /* 'rescale_matrix:10' assert( length(ts)>=ncols); */
  /* 'rescale_matrix:13' for ii=1:ncols */
  ii = 0;
  b_emxInit_real_T(&b_V, 1);
  while (ii + 1 <= ncols) {
    /* 'rescale_matrix:14' v = V(:,ii); */
    /* 'rescale_matrix:15' ts(ii) = norm2_vec(v); */
    kk = b_V->size[0];
    b_V->size[0] = V->size[0];
    emxEnsureCapacity((emxArray__common *)b_V, kk, (int32_T)sizeof(real_T));
    loop_ub = V->size[0] - 1;
    for (kk = 0; kk <= loop_ub; kk++) {
      b_V->data[kk] = V->data[kk + V->size[0] * ii];
    }

    ts->data[ii] = norm2_vec(b_V);

    /* 'rescale_matrix:17' if abs(ts(ii)) == 0 */
    if (fabs(ts->data[ii]) == 0.0) {
      /* 'rescale_matrix:18' ts(ii)=1; */
      ts->data[ii] = 1.0;
    } else {
      /* 'rescale_matrix:19' else */
      /* 'rescale_matrix:20' for kk=1:int32(size(V,1)) */
      loop_ub = V->size[0];
      for (kk = 0; kk + 1 <= loop_ub; kk++) {
        /* 'rescale_matrix:21' V(kk,ii) = V(kk,ii) / ts(ii); */
        V->data[kk + V->size[0] * ii] /= ts->data[ii];
      }
    }

    ii++;
  }

  emxFree_real_T(&b_V);
}

/*
 *
 */
static real_T sum(const emxArray_real_T *x)
{
  real_T y;
  int32_T vlen;
  int32_T k;
  if (x->size[0] == 0) {
    y = 0.0;
  } else {
    vlen = x->size[0];
    y = x->data[0];
    for (k = 2; k <= vlen; k++) {
      y += x->data[k - 1];
    }
  }

  return y;
}

/*
 * function [nrms, curs, prdirs] = compute_diffops_surf_cleanmesh(nv_clean,...
 *     xs, tris, nrms_proj, degree, ring, iterfit, nrms, curs, prdirs)
 */
void compute_diffops_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T *xs,
  const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, int32_T degree,
  real_T ring, boolean_T iterfit, emxArray_real_T *nrms, emxArray_real_T *curs,
  emxArray_real_T *prdirs)
{
  int32_T i1;
  uint32_T uv0[2];
  emxArray_int32_T *opphes;
  emxArray_int32_T *v2he;
  real_T u1;

  /* COMPUTE_DIFFOP_SURF_PARALLEL Compute differential operators on the */
  /* interior and boundary points of a submesh on a processor. */
  /* # coder.typeof( int32(0), [inf,3], [1,0]),coder.typeof( double(0), [inf,3], [1,0]), */
  /* # int32(0), double(0), true, coder.typeof( double(0), [inf,3], [1,0]), */
  /* # coder.typeof( double(0), [inf,2], [1,0]), */
  /* # coder.typeof( double(0), [inf,3], [1,0])} */
  /* 'compute_diffops_surf_cleanmesh:12' if nargin<6 */
  /* 'compute_diffops_surf_cleanmesh:13' if nargin<7 */
  /* 'compute_diffops_surf_cleanmesh:14' if nargin<8 && nargout>1 */
  /* 'compute_diffops_surf_cleanmesh:15' if nargin<9 && nargout>1 */
  /* 'compute_diffops_surf_cleanmesh:18' degree = max(1,min(6,degree)); */
  if (6 > degree) {
  } else {
    degree = 6;
  }

  if (1 < degree) {
  } else {
    degree = 1;
  }

  /* 'compute_diffops_surf_cleanmesh:19' if ring<=0 */
  if (ring <= 0.0) {
    /* 'compute_diffops_surf_cleanmesh:19' ring = 0.5*(double(degree) + 1); */
    ring = 0.5 * ((real_T)degree + 1.0);
  }

  /* 'compute_diffops_surf_cleanmesh:20' ring = max(1,min(3.5,ring)); */
  /*  Determine opposite halfedges */
  /* 'compute_diffops_surf_cleanmesh:22' opphes = coder.nullcopy(zeros(size(tris),'int32')); */
  for (i1 = 0; i1 < 2; i1++) {
    uv0[i1] = (uint32_T)tris->size[i1];
  }

  emxInit_int32_T(&opphes, 2);
  b_emxInit_int32_T(&v2he, 1);
  i1 = opphes->size[0] * opphes->size[1];
  opphes->size[0] = (int32_T)uv0[0];
  opphes->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)opphes, i1, (int32_T)sizeof(int32_T));

  /* 'compute_diffops_surf_cleanmesh:23' opphes = determine_opposite_halfedge_tri(int32(size(xs,1)), tris, opphes); */
  determine_opposite_halfedge_tri(xs->size[0], tris, opphes);

  /*  Determine incident halfedge. */
  /* 'compute_diffops_surf_cleanmesh:26' v2he = coder.nullcopy(zeros( size(xs,1),1,'int32')); */
  i1 = v2he->size[0];
  v2he->size[0] = xs->size[0];
  emxEnsureCapacity((emxArray__common *)v2he, i1, (int32_T)sizeof(int32_T));

  /* 'compute_diffops_surf_cleanmesh:27' v2he = determine_incident_halfedges( tris, opphes, v2he); */
  determine_incident_halfedges(tris, opphes, v2he);

  /*  Invoke fitting algorithm. Do not use iterative fitting except for linear */
  /*  fitting. Do not use interp point. */
  /* 'compute_diffops_surf_cleanmesh:32' if nargin<8 && nargout<2 */
  /* 'compute_diffops_surf_cleanmesh:35' else */
  /* 'compute_diffops_surf_cleanmesh:36' [nrms,curs,prdirs] = polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, ... */
  /* 'compute_diffops_surf_cleanmesh:37'         nrms_proj, opphes, v2he, degree, ring, iterfit, true, nrms, curs, prdirs); */
  u1 = 3.5 <= ring ? 3.5 : ring;
  u1 = 1.0 >= u1 ? 1.0 : u1;
  polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, nrms_proj, opphes, v2he, degree,
    u1, iterfit, nrms, curs, prdirs);
  emxFree_int32_T(&v2he);
  emxFree_int32_T(&opphes);
}

void compute_diffops_surf_cleanmesh_initialize(void)
{
}

void compute_diffops_surf_cleanmesh_terminate(void)
{
  /* (no terminate code required) */
}

emxArray_int32_T *emxCreateND_int32_T(int32_T numDimensions, int32_T *size)
{
  emxArray_int32_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  c_emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (int32_T *)calloc((uint32_T)numEl, sizeof(int32_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_real_T *emxCreateND_real_T(int32_T numDimensions, int32_T *size)
{
  emxArray_real_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  c_emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (real_T *)calloc((uint32_T)numEl, sizeof(real_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_int32_T *emxCreateWrapperND_int32_T(int32_T *data, int32_T
  numDimensions, int32_T *size)
{
  emxArray_int32_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  c_emxInit_int32_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_real_T *emxCreateWrapperND_real_T(real_T *data, int32_T numDimensions,
  int32_T *size)
{
  emxArray_real_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  c_emxInit_real_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_int32_T *emxCreateWrapper_int32_T(int32_T *data, int32_T rows, int32_T
  cols)
{
  emxArray_int32_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  c_emxInit_int32_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_real_T *emxCreateWrapper_real_T(real_T *data, int32_T rows, int32_T
  cols)
{
  emxArray_real_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  c_emxInit_real_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = data;
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  emx->canFreeData = FALSE;
  return emx;
}

emxArray_int32_T *emxCreate_int32_T(int32_T rows, int32_T cols)
{
  emxArray_int32_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  c_emxInit_int32_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (int32_T *)calloc((uint32_T)numEl, sizeof(int32_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

emxArray_real_T *emxCreate_real_T(int32_T rows, int32_T cols)
{
  emxArray_real_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  c_emxInit_real_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (real_T *)calloc((uint32_T)numEl, sizeof(real_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

void emxDestroyArray_int32_T(emxArray_int32_T *emxArray)
{
  emxFree_int32_T(&emxArray);
}

void emxDestroyArray_real_T(emxArray_real_T *emxArray)
{
  emxFree_real_T(&emxArray);
}

