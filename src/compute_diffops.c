#include "util.h"

static void average_vertex_normal_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *nrms);
static void b_abs(const real_T x[3], real_T y[3]);
static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static void b_eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3]);
static void b_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws);
static void b_fix(real_T *x);
static int32_T b_min(const emxArray_int32_T *varargin_1);
static void b_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2]);
static void b_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, const emxArray_int32_T *degs, int32_T
  degree, real_T ring, emxArray_real_T *curs);
static void b_polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2], real_T maxprdir[3]);
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
                      cend, const emxArray_real_T *ws);
static int32_T c_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T
  *bs, const emxArray_real_T *ws);
static void c_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2], real_T
  maxprdir[3]);
static void c_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T ring,
  emxArray_real_T *curs, emxArray_real_T *prdirs);
static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols);
static void d_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, const emxArray_int32_T *degs, int32_T
  degree, real_T ring, emxArray_real_T *curs, emxArray_real_T *prdirs);
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
static int32_T b_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags);
static void polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128], int32_T
  nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor, int32_T
  degree, real_T nrm[3], int32_T *deg);
static void polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T ring,
  emxArray_real_T *curs);
static void polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2]);
static int32_T qr_safeguarded(emxArray_real_T *A, int32_T ncols, emxArray_real_T
  *D);
static void rescale_matrix(emxArray_real_T *V, int32_T ncols, emxArray_real_T
  *ts);
static real_T sum(const emxArray_real_T *x);


/* Function Definitions */
static void average_vertex_normal_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *nrms)
{
  int32_T ntris;
  int32_T nv;
  int32_T i0;
  int32_T ix;
  int32_T ii;
  int32_T iy;
  real_T a[3];
  real_T b[3];
  real_T nrm[3];
  int32_T k;
  real_T y;

  /* AVERAGE_VERTEX_NORMAL_TRI Compute average vertex normal for surface  */
  /*  triangulation. */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT,FLABEL,OPPHES) Computes the average  */
  /*  vertex normal for surface triangulation, provided vertices XS, triangles */
  /*  TRIS, weighting options OPT, face-labels FLABEL and opposite vertices in  */
  /*  mx3 OPPHES. */
  /*   */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS) Same as above. See below for default  */
  /*  OPT and FLABEL values. */
  /*  */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT) Same as above. See below for  */
  /*  default FLABEL values. */
  /*  */
  /*  AVERAGE_VERTEX_NORMAL_TRI(XS,TRIS,OPT,FLABEL) Same as above. */
  /*      */
  /*  Set OPT to 'area', 'unit', 'angle', 'sphere', and 'bisect'. Default is  */
  /*  'area'. If FLABEL is given, then only faces with zero labels are  */
  /*  considered. */
  /*  */
  /*  See also AVERAGE_VERTEX_NORMAL_CURV */
  ntris = tris->size[0];
  nv = xs->size[0];
  i0 = nrms->size[0] * nrms->size[1];
  nrms->size[0] = nv;
  nrms->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)nrms, i0, (int32_T)sizeof(real_T));
  ix = nv * 3 - 1;
  for (i0 = 0; i0 <= ix; i0++) {
    nrms->data[i0] = 0.0;
  }

  for (ii = 0; ii + 1 <= ntris; ii++) {
    ix = tris->data[ii + (tris->size[0] << 1)];
    iy = tris->data[ii + tris->size[0]];
    for (i0 = 0; i0 < 3; i0++) {
      a[i0] = xs->data[(ix + xs->size[0] * i0) - 1] - xs->data[(iy + xs->size[0]
        * i0) - 1];
    }

    ix = tris->data[ii];
    iy = tris->data[ii + (tris->size[0] << 1)];
    for (i0 = 0; i0 < 3; i0++) {
      b[i0] = xs->data[(ix + xs->size[0] * i0) - 1] - xs->data[(iy + xs->size[0]
        * i0) - 1];
    }

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    nrm[0] = a[1] * b[2] - a[2] * b[1];
    nrm[1] = a[2] * b[0] - a[0] * b[2];
    nrm[2] = a[0] * b[1] - a[1] * b[0];
    for (k = 0; k < 3; k++) {
      ix = tris->data[ii + tris->size[0] * k];
      iy = tris->data[ii + tris->size[0] * k];
      for (i0 = 0; i0 < 3; i0++) {
        a[i0] = nrms->data[(iy + nrms->size[0] * i0) - 1] + nrm[i0];
      }

      for (i0 = 0; i0 < 3; i0++) {
        nrms->data[(ix + nrms->size[0] * i0) - 1] = a[i0];
      }
    }
  }

  for (ii = 0; ii + 1 <= nv; ii++) {
    for (i0 = 0; i0 < 3; i0++) {
      nrm[i0] = nrms->data[ii + nrms->size[0] * i0];
    }

    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrms->data[ii + nrms->size[0] * ix] * nrm[iy];
      ix++;
      iy++;
    }

    y = sqrt(y + 1.0E-100);
    for (i0 = 0; i0 < 3; i0++) {
      nrms->data[ii + nrms->size[0] * i0] /= y;
    }
  }
}

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
  grad_sqnorm = grad[0];
  y = pow(grad_sqnorm, 2.0);
  grad_sqnorm = grad[1];
  tmp = pow(grad_sqnorm, 2.0);
  grad_sqnorm = y + tmp;
  grad_norm = sqrt(grad_sqnorm);

  /*  Compute key parameters */
  ell = sqrt(1.0 + grad_sqnorm);
  if (grad_norm == 0.0) {
    c = 1.0;
    s = 0.0;
  } else {
    c = grad[0] / grad_norm;
    s = grad[1] / grad_norm;
  }

  /*  Compute mean curvature and Gaussian curvature */
  /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
  /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
  /*  Solve quadratic equation to compute principal curvatures */
  v[0] = c * H[0] + s * H[2];
  v[1] = c * H[2] + s * H[3];
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
  grad_sqnorm = W[0] + W[3];
  tmp = sqrt((W[0] - W[3]) * (W[0] - W[3]) + 4.0 * W[2] * W[2]);
  if (grad_sqnorm > 0.0) {
    curvs[0] = 0.5 * (grad_sqnorm + tmp);
    curvs[1] = 0.5 * (grad_sqnorm - tmp);
  } else {
    curvs[0] = 0.5 * (grad_sqnorm - tmp);
    curvs[1] = 0.5 * (grad_sqnorm + tmp);
  }

  /*  Compute principal directions, first with basis of left  */
  /*  singular vectors of Jacobian */
  /*  Compute principal directions in 3-D space */
  U[0] = c / ell;
  U[3] = -s;
  U[1] = s / ell;
  U[4] = c;
  U[2] = grad_norm / ell;
  U[5] = 0.0;
  if (curvs[0] == curvs[1]) {
    for (ix = 0; ix < 3; ix++) {
      dir[ix] = U[ix];
    }
  } else {
    if (fabs(W[0] - curvs[1]) > fabs(W[0] - curvs[0])) {
      d1[0] = W[0] - curvs[1];
      d1[1] = W[2];
    } else {
      d1[0] = -W[2];
      d1[1] = W[0] - curvs[0];
    }

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
}

static void b_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws)
{
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T nrow;
  int32_T k;
  emxArray_real_T *b_V;
  int32_T c_V;
  int32_T i8;
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
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC)  */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*   */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /*  Determine degree of fitting */
  npnts = us->size[0];

  /*  Determine degree of polynomial */
  ncols = (*degree + 2) * (*degree + 1) / 2 - 1;
  while ((npnts < ncols) && (*degree > 1)) {
    (*degree)--;
    ncols = (*degree + 2) * (*degree + 1) / 2 - 1;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  gen_vander_bivar(us, *degree, V);
  nrow = V->size[1];
  if (2 > nrow) {
    k = 0;
    nrow = 0;
  } else {
    k = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i8 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = nrow - k;
  emxEnsureCapacity((emxArray__common *)b_V, i8, (int32_T)sizeof(real_T));
  loop_ub = (nrow - k) - 1;
  for (nrow = 0; nrow <= loop_ub; nrow++) {
    b_loop_ub = c_V - 1;
    for (i8 = 0; i8 <= b_loop_ub; i8++) {
      b_V->data[i8 + b_V->size[0] * nrow] = V->data[i8 + V->size[0] * (k + nrow)];
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
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      nrow = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (nrow = 0; nrow <= loop_ub; nrow++) {
        ws1->data[nrow] = us->data[nrow] * us->data[nrow] + us->data[nrow +
          us->size[0]] * us->data[nrow + us->size[0]];
      }

      t2 = sum(ws1);
      t2 = t2 / (real_T)npnts * 0.01;
      nrow = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (nrow = 0; nrow <= loop_ub; nrow++) {
        ws1->data[nrow] += t2;
      }

      if (*degree < 4) {
        for (c_V = 0; c_V + 1 <= npnts; c_V++) {
          if (ws1->data[c_V] != 0.0) {
            ws1->data[c_V] = ws->data[c_V] / sqrt(ws1->data[c_V]);
          } else {
            ws1->data[c_V] = ws->data[c_V];
          }
        }
      } else {
        for (c_V = 0; c_V + 1 <= npnts; c_V++) {
          if (ws1->data[c_V] != 0.0) {
            ws1->data[c_V] = ws->data[c_V] / ws1->data[c_V];
          } else {
            ws1->data[c_V] = ws->data[c_V];
          }
        }
      }

      for (c_V = 0; c_V + 1 <= npnts; c_V++) {
        for (nrow = 0; nrow + 1 <= ncols; nrow++) {
          V->data[c_V + V->size[0] * nrow] *= ws1->data[c_V];
        }

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
      for (c_V = 0; c_V + 1 <= npnts; c_V++) {
        for (nrow = 0; nrow + 1 <= ncols; nrow++) {
          V->data[c_V + V->size[0] * nrow] *= ws->data[c_V];
        }

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
  nrow = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, nrow, (int32_T)sizeof(real_T));
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  nrow = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, nrow, (int32_T)sizeof(real_T));
  nrow = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  do {
    exitg1 = 0U;
    if (nrow < ncols) {
      (*degree)--;
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        nrow = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, nrow, (int32_T)sizeof(real_T));
        loop_ub = bs->size[0] - 1;
        for (nrow = 0; nrow <= loop_ub; nrow++) {
          bs->data[nrow] = 0.0;
        }

        exitg1 = 1U;
      } else {
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U) - 1;
      }
    } else {
      /* % Compute Q'bs */
      nrow = V->size[0];
      for (k = 0; k + 1 <= ncols; k++) {
        /*  Optimized version for */
        /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
        /*  where v is Q(k:npngs) */
        t2 = 0.0;
        for (c_V = k; c_V + 1 <= nrow; c_V++) {
          h_bs[0] = bs->size[0];
          h_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&h_bs;
          d_bs.numDimensions = 1;
          t2 += V->data[c_V + V->size[0] * k] * d_bs.data[c_V];
        }

        t2 += t2;
        for (c_V = k; c_V + 1 <= nrow; c_V++) {
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
      for (nrow = 0; nrow + 1 <= ncols; nrow++) {
        V->data[nrow + V->size[0] * nrow] = D->data[nrow];
      }

      /*  Perform backward substitution. */
      /*      bs = backsolve(R, bs) */
      /*      bs = backsolve(R, bs, cend) */
      /*      bs = backsolve(R, bs, cend, ws) */
      /*   Compute bs = (triu(R(1:cend,1:cend))\bs) ./ ws; */
      /*   The right-hand side vector bs can have multiple columns. */
      /*   By default, cend is size(R,2), and ws is ones. */
      for (nrow = ncols - 1; nrow + 1 > 0; nrow--) {
        for (c_V = nrow + 1; c_V + 1 <= ncols; c_V++) {
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

      /*  Scale bs back if ts is given. */
      for (nrow = 0; nrow + 1 <= ncols; nrow++) {
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

static void b_fix(real_T *x)
{
  if (*x > 0.0) {
    *x = floor(*x);
  } else {
    *x = ceil(*x);
  }
}

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

static void b_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2])
{
  int32_T i;
  int32_T ix;
  real_T absnrm[3];
  static const int8_T iv8[3] = { 0, 1, 0 };

  static const int8_T iv9[3] = { 1, 0, 0 };

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
  /* added */
  if (nverts == 0) {
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    *deg = 0;
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    b_abs(nrm, absnrm);
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv8[i];
      }
    } else {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv9[i];
      }
    }

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

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    ws_row->data[0] = 1.0;
    for (i = 0; i + 1 <= nverts; i++) {
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;
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

    if (degree == 0) {
      /*  Use linear fitting without weight */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      degree = 1;
    }

    /*  Compute the coefficients */
    *deg = degree;
    b_eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];
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
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
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
    if (*deg > 1) {
      H[0] = 2.0 * bs->data[2];
      H[2] = bs->data[3];
      H[1] = bs->data[3];
      H[3] = 2.0 * bs->data[4];
    } else {
      if (nverts >= 2) {
        if (*deg == 0) {
          emxInit_real_T(&b_us, 2);
          ix = b_us->size[0] * b_us->size[1];
          b_us->size[0] = 2;
          b_us->size[1] = 2;
          emxEnsureCapacity((emxArray__common *)b_us, ix, (int32_T)sizeof(real_T));
          for (ix = 0; ix < 2; ix++) {
            for (b_iy = 0; b_iy < 2; b_iy++) {
              b_us->data[b_iy + b_us->size[0] * ix] = us->data[b_iy + us->size[0]
                * ix];
            }
          }

          ix = us->size[0] * us->size[1];
          us->size[0] = b_us->size[0];
          us->size[1] = 2;
          emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
          for (ix = 0; ix < 2; ix++) {
            i = b_us->size[0] - 1;
            for (b_iy = 0; b_iy <= i; b_iy++) {
              us->data[b_iy + us->size[0] * ix] = b_us->data[b_iy + b_us->size[0]
                * ix];
            }
          }

          emxFree_real_T(&b_us);
          for (ix = 0; ix < 2; ix++) {
            ws_row->data[ix] = 1.0;
          }
        }

        /*  Try to compute curvatures from normals */
        linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row, cs2);
        H[0] = cs2[0];
        H[2] = cs2[1];
        H[1] = cs2[1];
        H[3] = cs2[2];
      }
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    if (*deg >= 1) {
      /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
      /* and pseudo-inverse. */
      /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
      /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
      /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
      /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
      /*  height function in 2x2 H with a local coordinate frame. */
      /*  */
      /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
      grad_norm = grad[1];
      y = pow(grad_norm, 2.0);
      grad_norm = grad[0];
      b_y = pow(grad_norm, 2.0);
      grad_sqnorm = b_y + y;
      grad_norm = sqrt(grad_sqnorm);

      /*  Compute key parameters */
      ell = sqrt(1.0 + grad_sqnorm);
      if (grad_norm == 0.0) {
        c = 1.0;
        s = 0.0;
      } else {
        c = grad[0] / grad_norm;
        s = grad[1] / grad_norm;
      }

      /*  Compute mean curvature and Gaussian curvature */
      /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
      /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
      /*  Solve quadratic equation to compute principal curvatures */
      b_v[0] = c * H[0] + s * H[2];
      b_v[1] = c * H[2] + s * H[3];
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
      grad_norm = H[0] + H[3];
      s = sqrt((H[0] - H[3]) * (H[0] - H[3]) + 4.0 * H[2] * H[2]);
      if (grad_norm > 0.0) {
        prcurvs[0] = 0.5 * (grad_norm + s);
        prcurvs[1] = 0.5 * (grad_norm - s);
      } else {
        prcurvs[0] = 0.5 * (grad_norm - s);
        prcurvs[1] = 0.5 * (grad_norm + s);
      }
    } else {
      for (i = 0; i < 2; i++) {
        prcurvs[i] = 0.0;
      }
    }
  }
}

static void b_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, const emxArray_int32_T *degs, int32_T
  degree, real_T ring, emxArray_real_T *curs)
{
  static const int8_T iv18[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T i5;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  int32_T ii;
  real_T ringv;
  int32_T deg_in;
  int32_T minpntsv;
  int32_T exitg1;
  int32_T ngbvs[128];
  real_T prcurvs[2];
  int32_T deg;

  /* POLYFIT_LHFGRAD_SURF Compute polynomial fitting of gradients with adaptive */
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
  /*  ring is double, as we allow half rings. */
  if (degree <= 6) {
    /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
    minpnts = (int32_T)iv18[degree - 1];
  } else {
    minpnts = 0;
  }

  emxInit_boolean_T(&vtags, 1);

  /*  Compute fitting at all vertices */
  nv = xs->size[0];
  i5 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i5, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i5 = 0; i5 <= loop_ub; i5++) {
    vtags->data[i5] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);
  i5 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i5, (int32_T)sizeof(boolean_T));
  loop_ub = tris->size[0] - 1;
  for (i5 = 0; i5 <= loop_ub; i5++) {
    ftags->data[i5] = FALSE;
  }

  for (ii = 1; ii <= nv; ii++) {
    /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
    if ((degs->size[0] > 1) && (degs->data[ii - 1] > 1)) {
    } else {
      ringv = ring;
      if ((degs->size[0] > 1) && (degs->data[ii - 1] == 0)) {
        /*  Use one-ring if degree is 0 */
        deg_in = 0;
        ringv = 1.0;
        minpntsv = 0;
      } else {
        deg_in = degree;
        minpntsv = minpnts;
      }

      do {
        exitg1 = 0U;

        /*  Collect neighbor vertices */
        loop_ub = b_obtain_nring_surf(ii, ringv, minpntsv, tris, opphes, v2he,
          ngbvs, vtags, ftags);
        polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, deg_in, &deg,
          prcurvs);
        if (curs->size[0] != 0) {
          for (i5 = 0; i5 < 2; i5++) {
            curs->data[(ii + curs->size[0] * i5) - 1] = prcurvs[i5];
          }
        }

        /*  Enlarge the neighborhood if necessary */
        if ((deg < deg_in) && (ringv < ring + ring)) {
          ringv += 0.5;
        } else {
          exitg1 = 1U;
        }
      } while (exitg1 == 0U);
    }
  }

  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

static void b_polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2], real_T maxprdir[3])
{
  int32_T i;
  int32_T iy;
  real_T nrm[3];
  int32_T k;
  real_T absnrm[3];
  static const int8_T iv4[3] = { 0, 1, 0 };

  static const int8_T iv5[3] = { 1, 0, 0 };

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
  if (nverts == 0) {
    *deg = 0;
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    for (i = 0; i < 3; i++) {
      maxprdir[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      nverts = 127;
    }

    /*  First, compute the rotation matrix */
    for (iy = 0; iy < 3; iy++) {
      nrm[iy] = nrms->data[(v + nrms->size[0] * iy) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    for (k = 0; k < 3; k++) {
      absnrm[k] = fabs(nrm[k]);
    }

    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv4[i];
      }
    } else {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv5[i];
      }
    }

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

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Evaluate local coordinate system and weights */
    iy = us->size[0] * us->size[1];
    us->size[0] = nverts + 1;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, iy, (int32_T)sizeof(real_T));
    iy = bs->size[0] * bs->size[1];
    bs->size[0] = nverts + 1;
    bs->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)bs, iy, (int32_T)sizeof(real_T));
    iy = ws_row->size[0];
    ws_row->size[0] = nverts + 1;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
    for (iy = 0; iy < 2; iy++) {
      us->data[us->size[0] * iy] = 0.0;
    }

    for (iy = 0; iy < 2; iy++) {
      bs->data[bs->size[0] * iy] = 0.0;
    }

    ws_row->data[0] = 1.0;
    for (i = 1; i <= nverts; i++) {
      for (iy = 0; iy < 3; iy++) {
        u[iy] = xs->data[(ngbvs[i - 1] + xs->size[0] * iy) - 1] - xs->data[(v +
          xs->size[0] * iy) - 1];
      }

      y = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        y += u[ix] * absnrm[b_iy];
        ix++;
        b_iy++;
      }

      us->data[i] = y;
      y = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        y += u[ix] * t2[b_iy];
        ix++;
        b_iy++;
      }

      us->data[i + us->size[0]] = y;
      h12 = 0.0;
      ix = 0;
      b_iy = 0;
      for (k = 0; k < 3; k++) {
        h12 += nrms->data[(ngbvs[i - 1] + nrms->size[0] * ix) - 1] * nrm[b_iy];
        ix++;
        b_iy++;
      }

      if (h12 > 0.0) {
        y = 0.0;
        ix = 0;
        b_iy = 0;
        for (k = 0; k < 3; k++) {
          y += nrms->data[(ngbvs[i - 1] + nrms->size[0] * ix) - 1] * absnrm[b_iy];
          ix++;
          b_iy++;
        }

        bs->data[i] = -y / h12;
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

      y = 0.0 >= h12 ? 0.0 : h12;
      ws_row->data[i] = y;
    }

    if (degree == 0) {
      /*  Use linear fitting without weight */
      i = ws_row->size[0];
      iy = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
      i--;
      for (iy = 0; iy <= i; iy++) {
        ws_row->data[iy] = 1.0;
      }

      degree = 1;
    }

    /*  Compute the coefficients and store them */
    *deg = degree;
    eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    grad[0] = bs->data[0];
    grad[1] = bs->data[bs->size[0]];
    h12 = bs->data[2] + bs->data[1 + bs->size[0]];
    h12 *= 0.5;
    H[0] = bs->data[1];
    H[2] = h12;
    H[1] = h12;
    H[3] = bs->data[2 + bs->size[0]];
    eval_curvature_lhf_surf(grad, H, prcurvs, u);

    /*  maxprdir = P * maxprdir_l; */
    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    for (iy = 0; iy < 3; iy++) {
      P[iy] = absnrm[iy];
      P[3 + iy] = t2[iy];
      P[6 + iy] = nrm[iy];
    }

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
  for (kk = 0; kk < 2; kk++) {
    for (jj = cend - 1; jj + 1 > 0; jj--) {
      for (ii = jj + 1; ii + 1 <= cend; ii++) {
        bs->data[jj + bs->size[0] * kk] -= R->data[jj + R->size[0] * ii] *
          bs->data[ii + bs->size[0] * kk];
      }

      bs->data[jj + bs->size[0] * kk] /= R->data[jj + R->size[0] * jj];
    }
  }

  /*  Scale bs back if ts is given. */
  for (kk = 0; kk < 2; kk++) {
    for (jj = 0; jj + 1 <= cend; jj++) {
      bs->data[jj + bs->size[0] * kk] /= ws->data[jj];
    }
  }
}

static int32_T c_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T
  *bs, const emxArray_real_T *ws)
{
  int32_T degree;
  emxArray_real_T *V;
  int32_T npnts;
  int32_T ii;
  int32_T jj;
  emxArray_real_T *b_V;
  int32_T c_V;
  int32_T i9;
  int32_T loop_ub;
  int32_T b_loop_ub;
  emxArray_real_T *ws1;
  emxArray_real_T *D;
  emxInit_real_T(&V, 2);

  /* EVAL_VANDER_BIVAR Evaluate generalized Vandermonde matrix. */
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC)  */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*   */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  degree = 1;

  /*  Determine degree of fitting */
  npnts = us->size[0];

  /*  Determine degree of polynomial */
  /* % Construct matrix */
  gen_vander_bivar(us, 1, V);
  ii = V->size[1];
  if (2 > ii) {
    jj = 0;
    ii = 0;
  } else {
    jj = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i9 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = ii - jj;
  emxEnsureCapacity((emxArray__common *)b_V, i9, (int32_T)sizeof(real_T));
  loop_ub = (ii - jj) - 1;
  for (ii = 0; ii <= loop_ub; ii++) {
    b_loop_ub = c_V - 1;
    for (i9 = 0; i9 <= b_loop_ub; i9++) {
      b_V->data[i9 + b_V->size[0] * ii] = V->data[i9 + V->size[0] * (jj + ii)];
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
  if (!(ws->size[0] == 0)) {
    for (ii = 0; ii + 1 <= npnts; ii++) {
      for (jj = 0; jj + 1 < 3; jj++) {
        V->data[ii + V->size[0] * jj] *= ws->data[ii];
      }

      for (jj = 0; jj < 2; jj++) {
        bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
      }
    }
  }

  b_emxInit_real_T(&ws1, 1);
  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  ii = ws1->size[0];
  ws1->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));
  rescale_matrix(V, 2, ws1);

  /* % Perform Householder QR factorization */
  ii = D->size[0];
  D->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)D, ii, (int32_T)sizeof(real_T));
  ii = qr_safeguarded(V, 2, D);

  /* % Adjust degree of fitting */
  if (ii < 2) {
    degree = 0;

    /*  Matrix is singular. Consider surface as flat. */
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
    compute_qtb(V, bs, 2);

    /* % Perform backward substitution and scale the solutions. */
    for (ii = 0; ii + 1 < 3; ii++) {
      V->data[ii + V->size[0] * ii] = D->data[ii];
    }

    backsolve(V, bs, 2, ws1);
  }

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
  return degree;
}

static void c_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2], real_T
  maxprdir[3])
{
  int32_T i;
  int32_T ix;
  real_T absnrm[3];
  static const int8_T iv10[3] = { 0, 1, 0 };

  static const int8_T iv11[3] = { 1, 0, 0 };

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
  /* added */
  if (nverts == 0) {
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    *deg = 0;
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }

    for (i = 0; i < 3; i++) {
      maxprdir[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    b_abs(nrm, absnrm);
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv10[i];
      }
    } else {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv11[i];
      }
    }

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

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    ws_row->data[0] = 1.0;
    for (i = 0; i + 1 <= nverts; i++) {
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;
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

    if (degree == 0) {
      /*  Use linear fitting without weight */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      degree = 1;
    }

    /*  Compute the coefficients */
    *deg = degree;
    b_eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];
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
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
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
    if (*deg > 1) {
      H[0] = 2.0 * bs->data[2];
      H[2] = bs->data[3];
      H[1] = bs->data[3];
      H[3] = 2.0 * bs->data[4];
    } else {
      if (nverts >= 2) {
        if (*deg == 0) {
          emxInit_real_T(&b_us, 2);
          ix = b_us->size[0] * b_us->size[1];
          b_us->size[0] = 2;
          b_us->size[1] = 2;
          emxEnsureCapacity((emxArray__common *)b_us, ix, (int32_T)sizeof(real_T));
          for (ix = 0; ix < 2; ix++) {
            for (b_iy = 0; b_iy < 2; b_iy++) {
              b_us->data[b_iy + b_us->size[0] * ix] = us->data[b_iy + us->size[0]
                * ix];
            }
          }

          ix = us->size[0] * us->size[1];
          us->size[0] = b_us->size[0];
          us->size[1] = 2;
          emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
          for (ix = 0; ix < 2; ix++) {
            i = b_us->size[0] - 1;
            for (b_iy = 0; b_iy <= i; b_iy++) {
              us->data[b_iy + us->size[0] * ix] = b_us->data[b_iy + b_us->size[0]
                * ix];
            }
          }

          emxFree_real_T(&b_us);
          for (ix = 0; ix < 2; ix++) {
            ws_row->data[ix] = 1.0;
          }
        }

        /*  Try to compute curvatures from normals */
        linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row, cs2);
        H[0] = cs2[0];
        H[2] = cs2[1];
        H[1] = cs2[1];
        H[3] = cs2[2];
      }
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    if (*deg >= 1) {
      b_eval_curvature_lhf_surf(grad, H, prcurvs, absnrm);

      /*  maxprdir = P * maxprdir_l; */
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
      for (i = 0; i < 2; i++) {
        prcurvs[i] = 0.0;
      }

      for (i = 0; i < 3; i++) {
        maxprdir[i] = 0.0;
      }
    }
  }
}

static void c_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T ring,
  emxArray_real_T *curs, emxArray_real_T *prdirs)
{
  static const int8_T iv19[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T i6;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  boolean_T b4;
  int32_T ii;
  real_T ringv;
  int32_T exitg1;
  int32_T ngbvs[128];
  real_T prcurvs[2];
  int32_T deg;
  real_T maxprdir[3];

  /* POLYFIT_LHFGRAD_SURF Compute polynomial fitting of gradients with adaptive */
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
  /*  ring is double, as we allow half rings. */
  if (degree <= 6) {
    /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
    minpnts = (int32_T)iv19[degree - 1];
  } else {
    minpnts = 0;
  }

  emxInit_boolean_T(&vtags, 1);

  /*  Compute fitting at all vertices */
  nv = xs->size[0];
  i6 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i6, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i6 = 0; i6 <= loop_ub; i6++) {
    vtags->data[i6] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);
  i6 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i6, (int32_T)sizeof(boolean_T));
  loop_ub = tris->size[0] - 1;
  for (i6 = 0; i6 <= loop_ub; i6++) {
    ftags->data[i6] = FALSE;
  }

  if (!(prdirs->size[0] != 0)) {
    b4 = TRUE;
  } else {
    b4 = FALSE;
  }

  for (ii = 1; ii <= nv; ii++) {
    /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
    ringv = ring;
    do {
      exitg1 = 0U;

      /*  Collect neighbor vertices */
      loop_ub = b_obtain_nring_surf(ii, ringv, minpnts, tris, opphes, v2he, ngbvs,
        vtags, ftags);
      if (b4) {
        polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, degree, &deg,
          prcurvs);
      } else {
        b_polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, degree, &deg,
          prcurvs, maxprdir);
        if (prdirs->size[0] != 0) {
          for (i6 = 0; i6 < 3; i6++) {
            prdirs->data[(ii + prdirs->size[0] * i6) - 1] = maxprdir[i6];
          }
        }
      }

      if (curs->size[0] != 0) {
        for (i6 = 0; i6 < 2; i6++) {
          curs->data[(ii + curs->size[0] * i6) - 1] = prcurvs[i6];
        }
      }

      /*  Enlarge the neighborhood if necessary */
      if ((deg < degree) && (ringv < ring + ring)) {
        ringv += 0.5;
      } else {
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols)
{
  int32_T nrow;
  int32_T k;
  int32_T jj;
  real_T t2;
  int32_T ii;
  nrow = Q->size[0];
  for (k = 0; k + 1 <= ncols; k++) {
    /*  Optimized version for */
    /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
    /*  where v is Q(k:npngs) */
    for (jj = 0; jj < 2; jj++) {
      t2 = 0.0;
      for (ii = k; ii + 1 <= nrow; ii++) {
        t2 += Q->data[ii + Q->size[0] * k] * bs->data[ii + bs->size[0] * jj];
      }

      t2 += t2;
      for (ii = k; ii + 1 <= nrow; ii++) {
        bs->data[ii + bs->size[0] * jj] -= t2 * Q->data[ii + Q->size[0] * k];
      }
    }
  }
}

static void d_polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, const emxArray_int32_T *degs, int32_T
  degree, real_T ring, emxArray_real_T *curs, emxArray_real_T *prdirs)
{
  static const int8_T iv20[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T i7;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  boolean_T b5;
  int32_T ii;
  real_T ringv;
  int32_T deg_in;
  int32_T minpntsv;
  int32_T exitg1;
  int32_T ngbvs[128];
  real_T prcurvs[2];
  int32_T deg;
  real_T maxprdir[3];

  /* POLYFIT_LHFGRAD_SURF Compute polynomial fitting of gradients with adaptive */
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
  /*  ring is double, as we allow half rings. */
  if (degree <= 6) {
    /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
    minpnts = (int32_T)iv20[degree - 1];
  } else {
    minpnts = 0;
  }

  emxInit_boolean_T(&vtags, 1);

  /*  Compute fitting at all vertices */
  nv = xs->size[0];
  i7 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i7, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    vtags->data[i7] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);
  i7 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i7, (int32_T)sizeof(boolean_T));
  loop_ub = tris->size[0] - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    ftags->data[i7] = FALSE;
  }

  if (!(prdirs->size[0] != 0)) {
    b5 = TRUE;
  } else {
    b5 = FALSE;
  }

  for (ii = 1; ii <= nv; ii++) {
    /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
    if ((degs->size[0] > 1) && (degs->data[ii - 1] > 1)) {
    } else {
      ringv = ring;
      if ((degs->size[0] > 1) && (degs->data[ii - 1] == 0)) {
        /*  Use one-ring if degree is 0 */
        deg_in = 0;
        ringv = 1.0;
        minpntsv = 0;
      } else {
        deg_in = degree;
        minpntsv = minpnts;
      }

      do {
        exitg1 = 0U;

        /*  Collect neighbor vertices */
        loop_ub = b_obtain_nring_surf(ii, ringv, minpntsv, tris, opphes, v2he,
          ngbvs, vtags, ftags);
        if (b5) {
          polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, deg_in, &deg,
            prcurvs);
        } else {
          b_polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, deg_in,
            &deg, prcurvs, maxprdir);
          if (prdirs->size[0] != 0) {
            for (i7 = 0; i7 < 3; i7++) {
              prdirs->data[(ii + prdirs->size[0] * i7) - 1] = maxprdir[i7];
            }
          }
        }

        if (curs->size[0] != 0) {
          for (i7 = 0; i7 < 2; i7++) {
            curs->data[(ii + curs->size[0] * i7) - 1] = prcurvs[i7];
          }
        }

        /*  Enlarge the neighborhood if necessary */
        if ((deg < deg_in) && (ringv < ring + ring)) {
          ringv += 0.5;
        } else {
          exitg1 = 1U;
        }
      } while (exitg1 == 0U);
    }
  }

  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

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
  grad_sqnorm = grad[0];
  y = pow(grad_sqnorm, 2.0);
  grad_sqnorm = grad[1];
  tmp = pow(grad_sqnorm, 2.0);
  grad_sqnorm = y + tmp;
  grad_norm = sqrt(grad_sqnorm);

  /*  Compute key parameters */
  ell = sqrt(1.0 + grad_sqnorm);
  if (grad_norm == 0.0) {
    c = 1.0;
    s = 0.0;
  } else {
    c = grad[0] / grad_norm;
    s = grad[1] / grad_norm;
  }

  /*  Compute mean curvature and Gaussian curvature */
  /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
  /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
  /*  Solve quadratic equation to compute principal curvatures */
  v[0] = c * H[0] + s * H[2];
  v[1] = c * H[2] + s * H[3];
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
  grad_sqnorm = W[0] + W[3];
  tmp = sqrt((W[0] - W[3]) * (W[0] - W[3]) + 4.0 * W[2] * W[2]);
  if (grad_sqnorm > 0.0) {
    curvs[0] = 0.5 * (grad_sqnorm + tmp);
    curvs[1] = 0.5 * (grad_sqnorm - tmp);
  } else {
    curvs[0] = 0.5 * (grad_sqnorm - tmp);
    curvs[1] = 0.5 * (grad_sqnorm + tmp);
  }

  /*  Compute principal directions, first with basis of left  */
  /*  singular vectors of Jacobian */
  /*  Compute principal directions in 3-D space */
  U[0] = c / ell;
  U[3] = -s;
  U[1] = s / ell;
  U[4] = c;
  U[2] = grad_norm / ell;
  U[5] = 0.0;
  if (curvs[0] == curvs[1]) {
    for (ix = 0; ix < 3; ix++) {
      dir[ix] = U[ix];
    }
  } else {
    if (fabs(W[0] - curvs[1]) > fabs(W[0] - curvs[0])) {
      d1[0] = W[0] - curvs[1];
      d1[1] = W[2];
    } else {
      d1[0] = -W[2];
      d1[1] = W[0] - curvs[0];
    }

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
}

static void eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
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
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC)  */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*   */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /*  Determine degree of fitting */
  npnts = us->size[0];

  /*  Determine degree of polynomial */
  ncols = (*degree + 2) * (*degree + 1) / 2;
  while ((npnts < ncols) && (*degree > 1)) {
    (*degree)--;
    ncols = (*degree + 2) * (*degree + 1) / 2;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  gen_vander_bivar(us, *degree, V);

  /* % Scale rows to assign different weights to different points */
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      jj = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] = us->data[jj] * us->data[jj] + us->data[jj + us->size[0]]
          * us->data[jj + us->size[0]];
      }

      A = sum(ws1);
      A = A / (real_T)npnts * 0.01;
      jj = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] += A;
      }

      if (*degree < 4) {
        for (ii = 0; ii + 1 <= npnts; ii++) {
          if (ws1->data[ii] != 0.0) {
            ws1->data[ii] = ws->data[ii] / sqrt(ws1->data[ii]);
          } else {
            ws1->data[ii] = ws->data[ii];
          }
        }
      } else {
        for (ii = 0; ii + 1 <= npnts; ii++) {
          if (ws1->data[ii] != 0.0) {
            ws1->data[ii] = ws->data[ii] / ws1->data[ii];
          } else {
            ws1->data[ii] = ws->data[ii];
          }
        }
      }

      for (ii = 0; ii + 1 <= npnts; ii++) {
        for (jj = 0; jj + 1 <= ncols; jj++) {
          V->data[ii + V->size[0] * jj] *= ws1->data[ii];
        }

        for (jj = 0; jj < 2; jj++) {
          bs->data[ii + bs->size[0] * jj] *= ws1->data[ii];
        }
      }
    } else {
      for (ii = 0; ii + 1 <= npnts; ii++) {
        for (jj = 0; jj + 1 <= ncols; jj++) {
          V->data[ii + V->size[0] * jj] *= ws->data[ii];
        }

        for (jj = 0; jj < 2; jj++) {
          bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
        }
      }
    }
  }

  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  jj = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  jj = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, jj, (int32_T)sizeof(real_T));
  ii = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  do {
    exitg1 = 0U;
    if (ii < ncols) {
      (*degree)--;
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
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
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U);
      }
    } else {
      /* % Compute Q'bs */
      compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      for (ii = 0; ii + 1 <= ncols; ii++) {
        V->data[ii + V->size[0] * ii] = D->data[ii];
      }

      backsolve(V, bs, ncols, ws1);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
}

static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V)
{
  int32_T npnts;
  emxArray_real_T *b_us;
  int32_T ncols;
  int32_T i1;
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
  static const int8_T iv0[10] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55 };

  emxArray_int32_T *r1;
  int8_T iv1[2];

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
  npnts = us->size[0];
  if (degree <= 0) {
    b_emxInit_real_T(&b_us, 1);
    degree = -degree;
    ncols = (1 + degree) * (1 + degree);
    i1 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = ncols;
    emxEnsureCapacity((emxArray__common *)V, i1, (int32_T)sizeof(real_T));

    /*  Preallocate storage */
    /*  Use tensor product */
    i1 = b_us->size[0];
    b_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)b_us, i1, (int32_T)sizeof(real_T));
    c = us->size[0] - 1;
    for (i1 = 0; i1 <= c; i1++) {
      b_us->data[i1] = us->data[i1];
    }

    emxInit_real_T(&v1, 2);
    b_emxInit_real_T(&c_us, 1);
    gen_vander_univar(b_us, degree, v1);
    i1 = c_us->size[0];
    c_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)c_us, i1, (int32_T)sizeof(real_T));
    emxFree_real_T(&b_us);
    c = us->size[0] - 1;
    for (i1 = 0; i1 <= c; i1++) {
      c_us->data[i1] = us->data[i1 + us->size[0]];
    }

    emxInit_real_T(&v2, 2);
    gen_vander_univar(c_us, degree, v2);
    p = 0;
    emxFree_real_T(&c_us);
    emxInit_real_T(&r0, 2);
    emxInit_real_T(&y, 2);
    b_emxInit_real_T(&a, 1);
    emxInit_real_T(&b_v2, 2);
    while (p + 1 <= npnts) {
      i1 = a->size[0];
      a->size[0] = v1->size[1];
      emxEnsureCapacity((emxArray__common *)a, i1, (int32_T)sizeof(real_T));
      c = v1->size[1] - 1;
      for (i1 = 0; i1 <= c; i1++) {
        a->data[i1] = v1->data[p + v1->size[0] * i1];
      }

      i1 = b_v2->size[0] * b_v2->size[1];
      b_v2->size[0] = 1;
      b_v2->size[1] = v2->size[1];
      emxEnsureCapacity((emxArray__common *)b_v2, i1, (int32_T)sizeof(real_T));
      c = v2->size[1] - 1;
      for (i1 = 0; i1 <= c; i1++) {
        b_v2->data[b_v2->size[0] * i1] = v2->data[p + v2->size[0] * i1];
      }

      i1 = y->size[0] * y->size[1];
      y->size[0] = a->size[0];
      y->size[1] = b_v2->size[1];
      emxEnsureCapacity((emxArray__common *)y, i1, (int32_T)sizeof(real_T));
      c = b_v2->size[1] - 1;
      for (i1 = 0; i1 <= c; i1++) {
        nx = a->size[0] - 1;
        for (kk2 = 0; kk2 <= nx; kk2++) {
          y->data[kk2 + y->size[0] * i1] = a->data[kk2] * b_v2->data[b_v2->size
            [0] * i1];
        }
      }

      nx = y->size[0] * y->size[1];
      for (i1 = 0; i1 < 2; i1++) {
        sz[i1] = 0;
      }

      sz[0] = 1;
      sz[1] = ncols;
      i1 = r0->size[0] * r0->size[1];
      r0->size[0] = 1;
      r0->size[1] = sz[1];
      emxEnsureCapacity((emxArray__common *)r0, i1, (int32_T)sizeof(real_T));
      for (c = 0; c + 1 <= nx; c++) {
        r0->data[c] = y->data[c];
      }

      c = r0->size[1] - 1;
      for (i1 = 0; i1 <= c; i1++) {
        V->data[p + V->size[0] * i1] = r0->data[r0->size[0] * i1];
      }

      p++;
    }

    emxFree_real_T(&b_v2);
    emxFree_real_T(&a);
    emxFree_real_T(&y);
    emxFree_real_T(&r0);
    emxFree_real_T(&v2);
    emxFree_real_T(&v1);
  } else {
    i1 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = (int32_T)iv0[degree];
    emxEnsureCapacity((emxArray__common *)V, i1, (int32_T)sizeof(real_T));
    c = npnts * iv0[degree] - 1;
    for (i1 = 0; i1 <= c; i1++) {
      V->data[i1] = 0.0;
    }

    /*  Preallocate storage */
    /*     %% Compute rows corresponding to function values */
    if (1 > npnts) {
      i1 = 0;
    } else {
      i1 = npnts;
    }

    b_emxInit_int32_T(&r1, 1);
    kk2 = r1->size[0];
    r1->size[0] = i1;
    emxEnsureCapacity((emxArray__common *)r1, kk2, (int32_T)sizeof(int32_T));
    c = i1 - 1;
    for (i1 = 0; i1 <= c; i1++) {
      r1->data[i1] = 1 + i1;
    }

    c = r1->size[0];
    emxFree_int32_T(&r1);
    c--;
    for (i1 = 0; i1 <= c; i1++) {
      V->data[i1] = 1.0;
    }

    for (i1 = 0; i1 < 2; i1++) {
      iv1[i1] = (int8_T)(i1 + 1);
    }

    for (i1 = 0; i1 < 2; i1++) {
      c = us->size[0] - 1;
      for (kk2 = 0; kk2 <= c; kk2++) {
        V->data[kk2 + V->size[0] * iv1[i1]] = us->data[kk2 + us->size[0] * i1];
      }
    }

    c = 3;
    for (nx = 2; nx <= degree; nx++) {
      for (kk2 = 1; kk2 <= nx; kk2++) {
        for (p = 0; p + 1 <= npnts; p++) {
          V->data[p + V->size[0] * c] = V->data[p + V->size[0] * (c - nx)] *
            us->data[p];
        }

        c++;
      }

      for (p = 0; p + 1 <= npnts; p++) {
        V->data[p + V->size[0] * c] = V->data[p + V->size[0] * ((c - nx) - 1)] *
          us->data[p + us->size[0]];
      }

      c++;
    }

    /*     %% Add rows corresponding to derivatives */
  }

  /*  Select subset of Vandermond matrix. */
  /*      V = subvander( V, npnts, rows, cols) */
}

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
  npnts = us->size[0];
  if (degree < 0) {
    degree = -degree;
  }

  ncols = degree + 1;
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

  if (degree > 0) {
    ncols = us->size[0];
    loop_ub = ncols - 1;
    for (p = 0; p <= loop_ub; p++) {
      V->data[p + V->size[0]] = us->data[p];
    }

    loop_ub = degree + 1;
    for (ncols = 1; ncols + 1 <= loop_ub; ncols++) {
      for (p = 0; p + 1 <= npnts; p++) {
        V->data[p + V->size[0] * ncols] = V->data[p + V->size[0] * (ncols - 1)] *
          us->data[p];
      }
    }
  }

  /* % Add rows corresponding to the derivatives multiplied by corresponding power of u */
  /*  Select subset of Vandermond matrix. */
  /*      V = subvander( V, npnts, rows, cols) */
}

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
  loop_ub = bs->size[0] * bs->size[1];
  bs->size[0] = us->size[0];
  bs->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)bs, loop_ub, (int32_T)sizeof(real_T));
  loop_ub = us->size[0];
  for (ii = 0; ii + 1 <= loop_ub; ii++) {
    if (ws->data[ii] > 0.0) {
      b = 0.0;
      ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        b += nrms->data[(ngbvs[ii] + nrms->size[0] * ix) - 1] * t1[iy];
        ix++;
        iy++;
      }

      bs->data[ii] = -b / ws->data[ii];
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
  c_eval_vander_bivar(us, bs, ws);
  b = bs->data[1] + bs->data[bs->size[0]];
  hess[0] = bs->data[0];
  hess[1] = 0.5 * b;
  hess[2] = bs->data[1 + bs->size[0]];
  emxFree_real_T(&bs);
}

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
  w = 0.0;
  for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
    u0 = fabs(v->data[(int32_T)ii - 1]);
    w = w >= u0 ? w : u0;
  }

  s = 0.0;
  if (w == 0.0) {
    /*  W can be zero for max(0,nan,...). Adding all three entries */
    /*  together will make sure NaN will be preserved. */
    for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
      s += v->data[(int32_T)ii - 1];
    }
  } else {
    for (ii = 1U; ii <= (uint32_T)v->size[0]; ii++) {
      u0 = v->data[(int32_T)ii - 1] / w;
      u0 = pow(u0, 2.0);
      s += u0;
    }

    s = w * sqrt(s);
  }

  return s;
}

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
  boolean_T b2;
  int32_T fid_in;
  static const int8_T iv16[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv17[3] = { 3, 1, 2 };

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
  boolean_T b3;
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
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  fid = (int32_T)((uint32_T)v2he->data[vid - 1] >> 2U) - 1;

  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  lid = (int32_T)((uint32_T)v2he->data[vid - 1] & 3U);
  nverts = 0;
  nfaces = 0;
  overflow = FALSE;
  if (!(fid + 1 != 0)) {
  } else {
    if ((ring == 1.0) && (minpnts == 0)) {
      b2 = TRUE;
    } else {
      b2 = FALSE;
    }

    /*  Optimized version for collecting one-ring vertices */
    if (opphes->data[fid + opphes->size[0] * lid] != 0) {
      fid_in = fid + 1;
    } else {
      fid_in = 0;
      nverts = 1;
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv16[lid] - 1)];
      if (!b2) {
        hebuf[0] = 0;
      }
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    do {
      exitg4 = 0U;

      /*  Insert vertx into list */
      lid = iv17[lid] - 1;
      if ((nverts < 128) && (nfaces < 256)) {
        nverts++;
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];
        if (!b2) {
          /*  Save starting position for next vertex */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv17[lid] -
            1)];
          nfaces++;
          ngbfs[nfaces - 1] = fid + 1;
        }
      } else {
        overflow = TRUE;
      }

      opp = opphes->data[fid + opphes->size[0] * lid];

      /*  HEID2FID   Obtains face ID from half-edge ID. */
      fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] * lid] >> 2U)
        - 1;
      if (fid + 1 == fid_in) {
        exitg4 = 1U;
      } else {
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        lid = (int32_T)((uint32_T)opp & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    if ((ring == 1.0) && ((nverts >= minpnts) || (nverts >= 128) || (nfaces >=
          256))) {
    } else {
      vtags->data[vid - 1] = TRUE;
      for (lid = 1; lid <= nverts; lid++) {
        vtags->data[ngbvs[lid - 1] - 1] = TRUE;
      }

      for (lid = 1; lid <= nfaces; lid++) {
        ftags->data[ngbfs[lid - 1] - 1] = TRUE;
      }

      /*  Define buffers and prepare tags for further processing */
      nverts_pre = 0;
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      ring_full = ring;
      b_fix(&ring_full);
      minpnts = minpnts <= 128 ? minpnts : 128;
      cur_ring = 1.0;
      do {
        exitg1 = 0U;
        guard1 = FALSE;
        if ((cur_ring > ring_full) || ((cur_ring == ring_full) && (ring_full !=
              ring))) {
          /*  Collect halfring */
          opp = nfaces;
          nverts_last = nverts;
          while (nfaces_pre + 1 <= opp) {
            /*  take opposite vertex in opposite face */
            lid = 0;
            exitg2 = 0U;
            while ((exitg2 == 0U) && (lid + 1 < 4)) {
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              fid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                opphes->size[0] * lid) - 1] >> 2U) - 1;
              if ((opphes->data[(ngbfs[nfaces_pre] + opphes->size[0] * lid) - 1]
                   != 0) && (!ftags->data[fid])) {
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  (iv17[lid] - 1)] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                if ((!ftags->data[fid]) && (!overflow)) {
                  nfaces++;
                  ngbfs[nfaces - 1] = fid + 1;
                  ftags->data[fid] = TRUE;
                }

                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv17[lid] -
                      1)] - 1]) && (!overflow)) {
                  nverts++;
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv17[lid]
                    - 1)];
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

          if ((nverts >= minpnts) || (nfaces >= 256) || (nfaces == opp)) {
            exitg1 = 1U;
          } else {
            /*  If needs to expand, then undo the last half ring */
            for (lid = nverts_last; lid + 1 <= nverts; lid++) {
              vtags->data[ngbvs[lid] - 1] = FALSE;
            }

            nverts = nverts_last;
            for (lid = opp; lid + 1 <= nfaces; lid++) {
              ftags->data[ngbfs[lid] - 1] = FALSE;
            }

            nfaces = opp;
            guard1 = TRUE;
          }
        } else {
          guard1 = TRUE;
        }

        if (guard1 == TRUE) {
          /*  Collect next full level of ring */
          nverts_last = nverts;
          nfaces_pre = nfaces;
          while (nverts_pre + 1 <= nverts_last) {
            /*  HEID2FID   Obtains face ID from half-edge ID. */
            fid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] >> 2U) -
              1;

            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            lid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] & 3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            if ((hebuf[nverts_pre] != 0) && (opphes->data[fid + opphes->size[0] *
                 lid] != 0)) {
              b3 = TRUE;
            } else {
              b3 = FALSE;
            }

            if (b3) {
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              fid = (int32_T)((uint32_T)hebuf[nverts_pre] >> 2U) - 1;

              /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
              lid = (int32_T)((uint32_T)hebuf[nverts_pre] & 3U);
            }

            /*  */
            if (opphes->data[fid + opphes->size[0] * lid] != 0) {
              fid_in = fid + 1;
            } else {
              fid_in = 0;
              if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                (iv16[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              if (!overflow) {
                nverts++;
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv16[lid]
                  - 1)];
                vtags->data[tris->data[fid + tris->size[0] * (iv16[lid] - 1)] -
                  1] = TRUE;

                /*  Save starting position for next vertex */
                hebuf[nverts - 1] = 0;
              }
            }

            /*  Rotate counterclockwise around the vertex. */
            isfirst = TRUE;
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              lid = iv17[lid] - 1;

              /*  Insert face into list */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                if (b3 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                if (overflow || ((!vtags->data[tris->data[fid + tris->size[0] *
                                  lid] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                if ((!vtags->data[tris->data[fid + tris->size[0] * lid] - 1]) &&
                    (!overflow)) {
                  nverts++;
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];
                  vtags->data[tris->data[fid + tris->size[0] * lid] - 1] = TRUE;

                  /*  Save starting position for next ring */
                  hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv17[lid] - 1)];
                }

                if ((!ftags->data[fid]) && (!overflow)) {
                  nfaces++;
                  ngbfs[nfaces - 1] = fid + 1;
                  ftags->data[fid] = TRUE;
                }

                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                opp = opphes->data[fid + opphes->size[0] * lid];

                /*  HEID2FID   Obtains face ID from half-edge ID. */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid] >> 2U) - 1;
                if (fid + 1 == fid_in) {
                  /*  Finished cycle */
                  exitg3 = 1U;
                } else {
                  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                  lid = (int32_T)((uint32_T)opp & 3U);
                }
              }
            } while (exitg3 == 0U);

            nverts_pre++;
          }

          cur_ring++;
          if (((nverts >= minpnts) && (cur_ring >= ring)) || (nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      vtags->data[vid - 1] = FALSE;
      for (lid = 1; lid <= nverts; lid++) {
        vtags->data[ngbvs[lid - 1] - 1] = FALSE;
      }

      if (!b2) {
        for (lid = 1; lid <= nfaces; lid++) {
          ftags->data[ngbfs[lid - 1] - 1] = FALSE;
        }
      }
    }
  }

  return nverts;
}

static void polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128], int32_T
  nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor, int32_T
  degree, real_T nrm[3], int32_T *deg)
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
  /* added */
  if (nverts == 0) {
    for (i = 0; i < 3; i++) {
      nrm[i] = 0.0;
    }

    *deg = 0;
  } else {
    if (nverts >= 128) {
      nverts = 127;
    }

    /*  First, determine local orthogonal cordinate system. */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms_coor->data[(v + nrms_coor->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    b_abs(nrm, absnrm);
    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv6[i];
      }
    } else {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv7[i];
      }
    }

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

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Project onto local coordinate system */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
    ix = bs->size[0];
    bs->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));
    ix = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    ws_row->data[0] = 1.0;
    for (i = 0; i + 1 <= nverts; i++) {
      for (ix = 0; ix < 3; ix++) {
        cs2[ix] = xs->data[(ngbvs[i] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * absnrm[iy];
        b_ix++;
        iy++;
      }

      us->data[i] = y;
      y = 0.0;
      b_ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        y += cs2[b_ix] * t2[iy];
        b_ix++;
        iy++;
      }

      us->data[i + us->size[0]] = y;
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

    if (degree == 0) {
      /*  Use linear fitting without weight */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      degree = 1;
    }

    /*  Compute the coefficients */
    *deg = degree;
    b_eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    grad[0] = bs->data[0];
    grad[1] = bs->data[1];
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
    for (ix = 0; ix < 3; ix++) {
      P[ix] = absnrm[ix];
      P[3 + ix] = t2[ix];
      P[6 + ix] = nrm[ix];
    }

    /*  nrm = P * nrm_l; */
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
    if ((*deg > 1) || (!(nverts >= 2))) {
    } else {
      if (*deg == 0) {
        emxInit_real_T(&b_us, 2);
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
        for (ix = 0; ix < 2; ix++) {
          ws_row->data[ix] = 1.0;
        }
      }

      /*  Try to compute curvatures from normals */
      linfit_lhf_grad_surf_point(ngbvs, us, absnrm, t2, nrms_coor, ws_row,
        unusedExpr);
    }

    emxFree_real_T(&ws_row);
    emxFree_real_T(&us);
  }
}

static void polyfit_lhfgrad_surf(const emxArray_real_T *xs, const
  emxArray_real_T *nrms, const emxArray_int32_T *tris, const emxArray_int32_T
  *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T ring,
  emxArray_real_T *curs)
{
  static const int8_T iv15[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T i4;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  int32_T ii;
  real_T ringv;
  int32_T exitg1;
  int32_T ngbvs[128];
  real_T prcurvs[2];
  int32_T deg;

  /* POLYFIT_LHFGRAD_SURF Compute polynomial fitting of gradients with adaptive */
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
  /*  ring is double, as we allow half rings. */
  if (degree <= 6) {
    /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
    minpnts = (int32_T)iv15[degree - 1];
  } else {
    minpnts = 0;
  }

  emxInit_boolean_T(&vtags, 1);

  /*  Compute fitting at all vertices */
  nv = xs->size[0];
  i4 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i4, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i4 = 0; i4 <= loop_ub; i4++) {
    vtags->data[i4] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);
  i4 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i4, (int32_T)sizeof(boolean_T));
  loop_ub = tris->size[0] - 1;
  for (i4 = 0; i4 <= loop_ub; i4++) {
    ftags->data[i4] = FALSE;
  }

  for (ii = 1; ii <= nv; ii++) {
    /*  If degs is nonempty, then only compute for vertices whose degree is 1 */
    ringv = ring;
    do {
      exitg1 = 0U;

      /*  Collect neighbor vertices */
      loop_ub = b_obtain_nring_surf(ii, ringv, minpnts, tris, opphes, v2he, ngbvs,
        vtags, ftags);
      polyfit_lhfgrad_surf_point(ii, ngbvs, loop_ub, xs, nrms, degree, &deg,
        prcurvs);
      if (curs->size[0] != 0) {
        for (i4 = 0; i4 < 2; i4++) {
          curs->data[(ii + curs->size[0] * i4) - 1] = prcurvs[i4];
        }
      }

      /*  Enlarge the neighborhood if necessary */
      if ((deg < degree) && (ringv < ring + ring)) {
        ringv += 0.5;
      } else {
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

static void polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2])
{
  int32_T i;
  int32_T ix;
  real_T nrm[3];
  real_T absnrm[3];
  static const int8_T iv2[3] = { 0, 1, 0 };

  static const int8_T iv3[3] = { 1, 0, 0 };

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
  if (nverts == 0) {
    *deg = 0;
    for (i = 0; i < 2; i++) {
      prcurvs[i] = 0.0;
    }
  } else {
    if (nverts >= 128) {
      nverts = 127;
    }

    /*  First, compute the rotation matrix */
    for (ix = 0; ix < 3; ix++) {
      nrm[ix] = nrms->data[(v + nrms->size[0] * ix) - 1];
    }

    /*  assert( 1.-nrm'*nrm < 1.e-10); */
    for (i = 0; i < 3; i++) {
      absnrm[i] = fabs(nrm[i]);
    }

    if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv2[i];
      }
    } else {
      for (i = 0; i < 3; i++) {
        absnrm[i] = (real_T)iv3[i];
      }
    }

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

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
    t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
    t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

    /*  Evaluate local coordinate system and weights */
    ix = us->size[0] * us->size[1];
    us->size[0] = nverts + 1;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));
    ix = bs->size[0] * bs->size[1];
    bs->size[0] = nverts + 1;
    bs->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)bs, ix, (int32_T)sizeof(real_T));
    ix = ws_row->size[0];
    ws_row->size[0] = nverts + 1;
    emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
    for (ix = 0; ix < 2; ix++) {
      us->data[us->size[0] * ix] = 0.0;
    }

    for (ix = 0; ix < 2; ix++) {
      bs->data[bs->size[0] * ix] = 0.0;
    }

    ws_row->data[0] = 1.0;
    for (ii = 1; ii <= nverts; ii++) {
      for (ix = 0; ix < 3; ix++) {
        u[ix] = xs->data[(ngbvs[ii - 1] + xs->size[0] * ix) - 1] - xs->data[(v +
          xs->size[0] * ix) - 1];
      }

      y = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        y += u[ix] * absnrm[iy];
        ix++;
        iy++;
      }

      us->data[ii] = y;
      y = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        y += u[ix] * t2[iy];
        ix++;
        iy++;
      }

      us->data[ii + us->size[0]] = y;
      h12 = 0.0;
      ix = 0;
      iy = 0;
      for (i = 0; i < 3; i++) {
        h12 += nrms->data[(ngbvs[ii - 1] + nrms->size[0] * ix) - 1] * nrm[iy];
        ix++;
        iy++;
      }

      if (h12 > 0.0) {
        y = 0.0;
        ix = 0;
        iy = 0;
        for (i = 0; i < 3; i++) {
          y += nrms->data[(ngbvs[ii - 1] + nrms->size[0] * ix) - 1] * absnrm[iy];
          ix++;
          iy++;
        }

        bs->data[ii] = -y / h12;
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

      y = 0.0 >= h12 ? 0.0 : h12;
      ws_row->data[ii] = y;
    }

    if (degree == 0) {
      /*  Use linear fitting without weight */
      i = ws_row->size[0];
      ix = ws_row->size[0];
      ws_row->size[0] = i;
      emxEnsureCapacity((emxArray__common *)ws_row, ix, (int32_T)sizeof(real_T));
      i--;
      for (ix = 0; ix <= i; ix++) {
        ws_row->data[ix] = 1.0;
      }

      degree = 1;
    }

    /*  Compute the coefficients and store them */
    *deg = degree;
    eval_vander_bivar(us, bs, deg, ws_row);

    /*  Convert coefficients into normals and curvatures */
    grad[0] = bs->data[0];
    grad[1] = bs->data[bs->size[0]];
    h12 = bs->data[2] + bs->data[1 + bs->size[0]];
    h12 *= 0.5;
    H[0] = bs->data[1];
    H[2] = h12;
    H[1] = h12;
    H[3] = bs->data[2 + bs->size[0]];

    /* EVAL_CURVATURE_LHF_SURF Compute principal curvature, principal direction  */
    /* and pseudo-inverse. */
    /*  [CURVS,DIR,JINV] = EVAL_CURVATURE_LHF_SURF(GRAD,H) Computes principal  */
    /*  curvature in 2x1 CURVS, principal direction of maximum curvature in 3x2  */
    /*  DIR, and pseudo-inverse of J in 2x3 JINV.  Input arguments are the */
    /*  gradient of the height function in 2x1 GRAD, and the Hessian of the */
    /*  height function in 2x2 H with a local coordinate frame. */
    /*  */
    /*  See also EVAL_CURVATURE_LHFINV_SURF, EVAL_CURVATURE_PARA_SURF */
    h12 = grad[0];
    y = pow(h12, 2.0);
    h12 = grad[1];
    h12 = pow(h12, 2.0);
    grad_sqnorm = y + h12;
    h12 = sqrt(grad_sqnorm);

    /*  Compute key parameters */
    ell = sqrt(1.0 + grad_sqnorm);
    emxFree_real_T(&ws_row);
    emxFree_real_T(&bs);
    emxFree_real_T(&us);
    if (h12 == 0.0) {
      c = 1.0;
      s = 0.0;
    } else {
      c = grad[0] / h12;
      s = grad[1] / h12;
    }

    /*  Compute mean curvature and Gaussian curvature */
    /*  kH2 = (H(1,1)+H(2,2))/ell - grad*H*grad'/ell3; */
    /*  kG =  (H(1,1)*H(2,2)-H(1,2)^2)/ell2^2; */
    /*  Solve quadratic equation to compute principal curvatures */
    grad[0] = c * H[0] + s * H[2];
    grad[1] = c * H[2] + s * H[3];
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
    h12 = H[0] + H[3];
    s = sqrt((H[0] - H[3]) * (H[0] - H[3]) + 4.0 * H[2] * H[2]);
    if (h12 > 0.0) {
      prcurvs[0] = 0.5 * (h12 + s);
      prcurvs[1] = 0.5 * (h12 - s);
    } else {
      prcurvs[0] = 0.5 * (h12 - s);
      prcurvs[1] = 0.5 * (h12 + s);
    }
  }
}

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
  rnk = ncols;
  nrows = A->size[0];
  jj = v->size[0];
  v->size[0] = nrows;
  emxEnsureCapacity((emxArray__common *)v, jj, (int32_T)sizeof(real_T));
  k = 0;
  exitg1 = 0U;
  while ((exitg1 == 0U) && (k + 1 <= ncols)) {
    nv = nrows - k;
    for (jj = 0; jj + 1 <= nv; jj++) {
      v->data[jj] = A->data[(jj + k) + A->size[0] * k];
    }

    /*  We don't need to worry about overflow, since A has been rescaled. */
    t2 = 0.0;
    for (jj = 0; jj + 1 <= nv; jj++) {
      t2 += v->data[jj] * v->data[jj];
    }

    t = sqrt(t2);
    if (v->data[0] >= 0.0) {
      t2 = sqrt(2.0 * (t2 + v->data[0] * t));
      v->data[0] += t;
    } else {
      t2 = sqrt(2.0 * (t2 - v->data[0] * t));
      v->data[0] -= t;
    }

    if (t2 > 0.0) {
      for (jj = 0; jj + 1 <= nv; jj++) {
        v->data[jj] /= t2;
      }
    }

    /*  Optimized version for */
    /*  A(k:npnts,k:ncols) = A(k:npnts,k:ncols) - 2*v*(v'*A(k:npnts,k:ncols)); */
    for (jj = k; jj + 1 <= ncols; jj++) {
      t2 = 0.0;
      for (ii = 0; ii + 1 <= nv; ii++) {
        t2 += v->data[ii] * A->data[(ii + k) + A->size[0] * jj];
      }

      t2 += t2;
      for (ii = 0; ii + 1 <= nv; ii++) {
        A->data[(ii + k) + A->size[0] * jj] -= t2 * v->data[ii];
      }
    }

    D->data[k] = A->data[k + A->size[0] * k];
    for (jj = 0; jj + 1 <= nv; jj++) {
      A->data[(jj + k) + A->size[0] * k] = v->data[jj];
    }

    /*  Estimate rank of matrix */
    if (fabs(D->data[k]) < 1.0E-8) {
      rnk = k;
      exitg1 = 1U;
    } else {
      k++;
    }
  }

  emxFree_real_T(&v);
  return rnk;
}

static void rescale_matrix(emxArray_real_T *V, int32_T ncols, emxArray_real_T
  *ts)
{
  int32_T ii;
  emxArray_real_T *b_V;
  int32_T kk;
  int32_T loop_ub;

  /* % Rescale the columns of a matrix to reduce condition number */
  ii = 0;
  b_emxInit_real_T(&b_V, 1);
  while (ii + 1 <= ncols) {
    kk = b_V->size[0];
    b_V->size[0] = V->size[0];
    emxEnsureCapacity((emxArray__common *)b_V, kk, (int32_T)sizeof(real_T));
    loop_ub = V->size[0] - 1;
    for (kk = 0; kk <= loop_ub; kk++) {
      b_V->data[kk] = V->data[kk + V->size[0] * ii];
    }

    ts->data[ii] = norm2_vec(b_V);
    if (fabs(ts->data[ii]) == 0.0) {
      ts->data[ii] = 1.0;
    } else {
      loop_ub = V->size[0];
      for (kk = 0; kk + 1 <= loop_ub; kk++) {
        V->data[kk + V->size[0] * ii] /= ts->data[ii];
      }
    }

    ii++;
  }

  emxFree_real_T(&b_V);
}

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


void compute_diffops_surf(const emxArray_real_T *xs, const emxArray_int32_T
  *tris, int32_T degree, real_T ring, boolean_T iterfit, emxArray_real_T *nrms,
  emxArray_real_T *curs, emxArray_real_T *prdirs, int32_T param)
{
  real_T ringv;
  int32_T i2;
  uint32_T uv0[2];
  emxArray_int32_T *opphes;
  emxArray_int32_T *v2he;
  emxArray_real_T *nrms_proj;
  emxArray_boolean_T *vtags;
  static const int8_T iv12[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  int32_T nv;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  emxArray_int32_T *degs;
  boolean_T b0;
  boolean_T b1;
  int32_T ii;
  int32_T exitg1;
  int32_T ngbvs[128];
  int32_T deg;
  real_T nrm[3];
  real_T prcurvs[2];
  real_T maxprdir[3];

  /* COMPUTE_DIFFOP_SURF Compute differential operators. */
  /*  [NRMS, CURS, PRDIRS] = COMPUTE_DIFFOPS_SURF( XS, TRIS, DEGREE, RING, ... */
  /*  ITERFIT, NRMS, CURS, PRDIRS, PARAM) computes differential operators, */
  /*  provided vertex coordinates in nx3 XS, element connectivity in mx3 TRIS, */
  /*  degree of fitting DEGREE (default is 2), normals in nx3 NRMS, principal */
  /*  curvatures in nx2 CURS, and principal directions in nx3 PRDIRS. */
  /*  */
  /*  See also COMPUTE_DIFFOPS_CURV */
  /* # coder.typeof( int32(0), [inf,3], [1,0]), int32(0), double(0), */
  /* # true, coder.typeof( double(0), [inf,3], [1,0]),  */
  /* # coder.typeof( double(0), [inf,2], [1,0]),  */
  /* # coder.typeof( double(0), [inf,3], [1,0]), int32(0)} */
  /*  Set param to 1 to enable conformal parameterization-based */
  /*  approach. Set param to 0 to use local height function. */
  if (6 > degree) {
  } else {
    degree = 6;
  }

  if (1 < degree) {
  } else {
    degree = 1;
  }

  if (ring <= 0.0) {
    ring = 0.5 * ((real_T)degree + 1.0);
  }

  ringv = 3.5 <= ring ? 3.5 : ring;
  ring = 1.0 >= ringv ? 1.0 : ringv;

  /*  Determine opposite halfedges */
  for (i2 = 0; i2 < 2; i2++) {
    uv0[i2] = (uint32_T)tris->size[i2];
  }

  emxInit_int32_T(&opphes, 2);
  b_emxInit_int32_T(&v2he, 1);
  emxInit_real_T(&nrms_proj, 2);
  emxInit_boolean_T(&vtags, 1);
  i2 = opphes->size[0] * opphes->size[1];
  opphes->size[0] = (int32_T)uv0[0];
  opphes->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)opphes, i2, (int32_T)sizeof(int32_T));
  determine_opposite_halfedge_tri(xs->size[0], tris, opphes);

  /*  Determine incident halfedge. */
  i2 = v2he->size[0];
  v2he->size[0] = xs->size[0];
  emxEnsureCapacity((emxArray__common *)v2he, i2, (int32_T)sizeof(int32_T));
  determine_incident_halfedges(tris, opphes, v2he);

  /*  Invoke fitting algorithm. Do not use iterative fitting except for linear */
  /*  fitting. Do not use interp point. */
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
  /* #   coder.typeof(int32(0),[inf,3],[1,0]), */
  /* #   coder.typeof(int32(0),[inf,1],[1,0]), */
  /* #   int32(0),0,false,false, */
  /* #   coder.typeof(0,[inf,3],[1,0]), */
  /* #   coder.typeof(0,[inf,2],[1,0]), */
  /* #   coder.typeof(0,[inf,3],[1,0])} */
  /*  ring is double, as we allow half rings. */
  /*  pntsneeded = [3 6 10 15 21 28]*1.5; */
  minpnts = (int32_T)iv12[degree - 1];

  /*  Compute average vertex normals for local projection. */
  average_vertex_normal_tri(xs, tris, nrms_proj);
  nv = xs->size[0];
  i2 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i2, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i2 = 0; i2 <= loop_ub; i2++) {
    vtags->data[i2] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);
  i2 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i2, (int32_T)sizeof(boolean_T));
  loop_ub = tris->size[0] - 1;
  for (i2 = 0; i2 <= loop_ub; i2++) {
    ftags->data[i2] = FALSE;
  }

  b_emxInit_int32_T(&degs, 1);
  i2 = degs->size[0];
  degs->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)degs, i2, (int32_T)sizeof(int32_T));
  b0 = (prdirs->size[0] == 0);
  if ((degree == 1) || (curs->size[0] == 0) || ((curs->size[0] == 0) &&
       (prdirs->size[0] == 0))) {
    b1 = TRUE;
  } else {
    b1 = FALSE;
  }

  for (ii = 1; ii <= nv; ii++) {
    ringv = ring;
    do {
      exitg1 = 0U;

      /*  Collect neighbor vertices */
      loop_ub = b_obtain_nring_surf(ii, ringv, minpnts, tris, opphes, v2he, ngbvs,
        vtags, ftags);
      if (b1) {
        polyfit_lhf_surf_point(ii, ngbvs, loop_ub, xs, nrms_proj, degree, nrm,
          &deg);
      } else {
        if (b0) {
          b_polyfit_lhf_surf_point(ii, ngbvs, loop_ub, xs, nrms_proj, degree,
            nrm, &deg, prcurvs);
        } else {
          c_polyfit_lhf_surf_point(ii, ngbvs, loop_ub, xs, nrms_proj, degree,
            nrm, &deg, prcurvs, maxprdir);
          if (prdirs->size[0] != 0) {
            for (i2 = 0; i2 < 3; i2++) {
              prdirs->data[(ii + prdirs->size[0] * i2) - 1] = maxprdir[i2];
            }
          }
        }

        if (curs->size[0] != 0) {
          for (i2 = 0; i2 < 2; i2++) {
            curs->data[(ii + curs->size[0] * i2) - 1] = prcurvs[i2];
          }
        }
      }

      degs->data[ii - 1] = deg;
      if (nrms->size[0] != 0) {
        for (i2 = 0; i2 < 3; i2++) {
          nrms->data[(ii + nrms->size[0] * i2) - 1] = nrm[i2];
        }
      }

      /*  Enlarge the neighborhood if necessary */
      if ((deg < degree) && (ringv < ring + ring)) {
        ringv += 0.5;

        /*  Enlarge the neighborhood */
      } else {
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
  emxFree_real_T(&nrms_proj);
  if ((!iterfit) && (!(curs->size[0] != 0)) && (!(prdirs->size[0] != 0))) {
  } else {
    /* % */
    if (prdirs->size[0] == 0) {
      if (iterfit) {
        polyfit_lhfgrad_surf(xs, nrms, tris, opphes, v2he, degree, ring, curs);
      } else {
        if (b_min(degs) <= 1) {
          b_polyfit_lhfgrad_surf(xs, nrms, tris, opphes, v2he, degs, degree,
            ring, curs);
        }
      }
    } else if (iterfit) {
      c_polyfit_lhfgrad_surf(xs, nrms, tris, opphes, v2he, degree, ring, curs,
        prdirs);
    } else {
      if (b_min(degs) <= 1) {
        d_polyfit_lhfgrad_surf(xs, nrms, tris, opphes, v2he, degs, degree, ring,
          curs, prdirs);
      }
    }
  }

  emxFree_int32_T(&degs);
  emxFree_int32_T(&v2he);
  emxFree_int32_T(&opphes);
}

