#include "util.h"

static void average_vertex_normal_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *nrms);
static void b_abs(const real_T x[3], real_T y[3]);
static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static int32_T backsolve_bivar_safeguarded(const emxArray_real_T *R,
  emxArray_real_T *bs, int32_T degree, const emxArray_real_T *ws);
static boolean_T compute_weights(const emxArray_real_T *us, const
  emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws);
static int32_T eval_vander_bivar_cmf(const emxArray_real_T *us, emxArray_real_T *
  bs, int32_T degree, const emxArray_real_T *ws);
static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static void gen_vander_univar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static int32_T mrdivide(int32_T A, real_T B);
static real_T norm2_vec(const emxArray_real_T *v);
static int32_T b_obtain_nring_surf(int32_T vid, int32_T ring, real_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, real_T ngbvs[128], emxArray_real_T *vtags,
  emxArray_real_T *ftags);
static void polyfit3d_walf_tri(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, const emxArray_real_T *ngbpnts3, const emxArray_real_T *nrms3, const
  emxArray_real_T *xi, const emxArray_real_T *eta, int32_T deg, emxArray_real_T *
  pnt);
static void polyfit3d_walf_vertex(const emxArray_real_T *pnts, const
  emxArray_real_T *nrms, const emxArray_real_T *pos, int32_T deg,
  emxArray_real_T *pnt);

/* Function Definitions */
static void average_vertex_normal_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *nrms)
{
  int32_T ntris;
  int32_T nv;
  int32_T i2;
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
  i2 = nrms->size[0] * nrms->size[1];
  nrms->size[0] = nv;
  nrms->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)nrms, i2, (int32_T)sizeof(real_T));
  ix = nv * 3 - 1;
  for (i2 = 0; i2 <= ix; i2++) {
    nrms->data[i2] = 0.0;
  }

  for (ii = 0; ii + 1 <= ntris; ii++) {
    ix = tris->data[ii + (tris->size[0] << 1)];
    iy = tris->data[ii + tris->size[0]];
    for (i2 = 0; i2 < 3; i2++) {
      a[i2] = xs->data[(ix + xs->size[0] * i2) - 1] - xs->data[(iy + xs->size[0]
        * i2) - 1];
    }

    ix = tris->data[ii];
    iy = tris->data[ii + (tris->size[0] << 1)];
    for (i2 = 0; i2 < 3; i2++) {
      b[i2] = xs->data[(ix + xs->size[0] * i2) - 1] - xs->data[(iy + xs->size[0]
        * i2) - 1];
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
      for (i2 = 0; i2 < 3; i2++) {
        a[i2] = nrms->data[(iy + nrms->size[0] * i2) - 1] + nrm[i2];
      }

      for (i2 = 0; i2 < 3; i2++) {
        nrms->data[(ix + nrms->size[0] * i2) - 1] = a[i2];
      }
    }
  }

  for (ii = 0; ii + 1 <= nv; ii++) {
    for (i2 = 0; i2 < 3; i2++) {
      nrm[i2] = nrms->data[ii + nrms->size[0] * i2];
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
    for (i2 = 0; i2 < 3; i2++) {
      nrms->data[ii + nrms->size[0] * i2] /= y;
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

static int32_T backsolve_bivar_safeguarded(const emxArray_real_T *R,
  emxArray_real_T *bs, int32_T degree, const emxArray_real_T *ws)
{
  int32_T deg_out;
  emxArray_real_T *bs_bak;
  int32_T ncols;
  int32_T ii;
  int32_T i;
  int32_T b_bs[2];
  emxArray_real_T c_bs;
  emxArray_real_T *tb;
  boolean_T exitg1;
  int32_T cend;
  boolean_T downgrade;
  int32_T d;
  boolean_T exitg2;
  int32_T cstart;
  int32_T d_bs[2];
  int32_T e_bs[2];
  int32_T f_bs[2];
  emxArray_real_T g_bs;
  emxArray_real_T h_bs;
  int32_T i_bs[2];
  int32_T j_bs[2];
  boolean_T guard1 = FALSE;
  int32_T jind;
  boolean_T exitg3;
  int32_T k_bs[2];
  real_T err;
  int32_T l_bs[2];
  int32_T m_bs[2];
  int32_T n_bs[2];
  int32_T o_bs[2];
  emxInit_real_T(&bs_bak, 1);

  /*  Perform back substitution with safeguards to downgrade the order if necessary. */
  /*      [bs,deg_out] = backsolve_bivar_safeguarded(R, bs, degree, interp, ws) */
  /*  Second, solve for each degree in decending order */
  deg_out = degree;
  ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
  ii = bs_bak->size[0];
  bs_bak->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)bs_bak, ii, (int32_T)sizeof(real_T));
  if (degree >= 2) {
    /*  Back up bs for computing reference solution and for resolving */
    /*  the backsolve after lowering degree. */
    for (i = 0; i + 1 <= ncols; i++) {
      b_bs[0] = bs->size[0];
      b_bs[1] = 1;
      c_bs = *bs;
      c_bs.size = (int32_T *)&b_bs;
      c_bs.numDimensions = 1;
      bs_bak->data[i] = c_bs.data[i];
    }
  }

  emxInit_real_T(&tb, 1);
  exitg1 = 0U;
  while ((exitg1 == 0U) && (deg_out >= 1)) {
    cend = ncols;
    downgrade = FALSE;
    d = deg_out;
    exitg2 = 0U;
    while ((exitg2 == 0U) && (d > -1)) {
      cstart = (int32_T)((uint32_T)(d * (d + 1)) >> 1U);

      /*  Solve for bs */
      for (i = cend - 1; i + 1 >= cstart + 1; i--) {
        for (ii = i + 1; ii + 1 <= ncols; ii++) {
          d_bs[0] = bs->size[0];
          d_bs[1] = 1;
          e_bs[0] = bs->size[0];
          e_bs[1] = 1;
          f_bs[0] = bs->size[0];
          f_bs[1] = 1;
          c_bs = *bs;
          c_bs.size = (int32_T *)&d_bs;
          c_bs.numDimensions = 1;
          g_bs = *bs;
          g_bs.size = (int32_T *)&e_bs;
          g_bs.numDimensions = 1;
          h_bs = *bs;
          h_bs.size = (int32_T *)&f_bs;
          h_bs.numDimensions = 1;
          c_bs.data[i] = g_bs.data[i] - R->data[i + R->size[0] * ii] *
            h_bs.data[ii];
        }

        i_bs[0] = bs->size[0];
        i_bs[1] = 1;
        j_bs[0] = bs->size[0];
        j_bs[1] = 1;
        c_bs = *bs;
        c_bs.size = (int32_T *)&i_bs;
        c_bs.numDimensions = 1;
        g_bs = *bs;
        g_bs.size = (int32_T *)&j_bs;
        g_bs.numDimensions = 1;
        c_bs.data[i] = g_bs.data[i] / R->data[i + R->size[0] * i];
      }

      /*  Check whether a coefficient has changed substantially by higher- */
      /*  order terms. If so, then decrease the degree of fitting. */
      guard1 = FALSE;
      if ((d >= 2) && (d < deg_out)) {
        if (cstart + 1 > cend) {
          ii = 0;
          jind = 0;
        } else {
          ii = cstart;
          jind = cend;
        }

        i = tb->size[0];
        tb->size[0] = jind - ii;
        emxEnsureCapacity((emxArray__common *)tb, i, (int32_T)sizeof(real_T));
        i = (jind - ii) - 1;
        for (jind = 0; jind <= i; jind++) {
          tb->data[jind] = bs_bak->data[ii + jind];
        }

        i = cend - 1;
        exitg3 = 0U;
        while ((exitg3 == 0U) && (i + 1 >= cstart + 1)) {
          jind = i - cstart;
          for (ii = i + 2; ii <= cend; ii++) {
            tb->data[jind] -= R->data[i + R->size[0] * (ii - 1)] * tb->data[(ii
              - cstart) - 1];
          }

          tb->data[jind] /= R->data[i + R->size[0] * i];
          k_bs[0] = bs->size[0];
          k_bs[1] = 1;
          c_bs = *bs;
          c_bs.size = (int32_T *)&k_bs;
          c_bs.numDimensions = 1;
          err = c_bs.data[i] - tb->data[jind];
          err = fabs(err);
          if ((err > 0.05) && (err >= 1.05 * fabs(tb->data[jind]))) {
            downgrade = TRUE;
            exitg3 = 1U;
          } else {
            i--;
          }
        }

        if (downgrade) {
          exitg2 = 1U;
        } else {
          guard1 = TRUE;
        }
      } else {
        guard1 = TRUE;
      }

      if (guard1 == TRUE) {
        cend = cstart;
        d--;
      }
    }

    if (!downgrade) {
      exitg1 = 1U;
    } else {
      /*  Decrease the degree of fitting by one. */
      /*  An alternative is to decreaes deg to d. This may be more */
      /*  efficient but it lose some chances to obtain higher-order accuracy. */
      deg_out--;
      ncols = (int32_T)((uint32_T)((deg_out + 2) * (deg_out + 1)) >> 1U);

      /*  Restore bs. */
      if (1 > ncols) {
        ii = -1;
      } else {
        ii = ncols - 1;
      }

      i = bs->size[0];
      l_bs[0] = i;
      l_bs[1] = 1;
      for (jind = 0; jind <= ii; jind++) {
        c_bs = *bs;
        c_bs.size = (int32_T *)&l_bs;
        c_bs.numDimensions = 1;
        c_bs.data[jind] = bs_bak->data[jind];
      }
    }
  }

  emxFree_real_T(&tb);
  emxFree_real_T(&bs_bak);

  /*  Done with the current right-hand-side column. */
  /*  Scale back bs. */
  for (i = 0; i + 1 <= ncols; i++) {
    m_bs[0] = bs->size[0];
    m_bs[1] = 1;
    n_bs[0] = bs->size[0];
    n_bs[1] = 1;
    c_bs = *bs;
    c_bs.size = (int32_T *)&m_bs;
    c_bs.numDimensions = 1;
    g_bs = *bs;
    g_bs.size = (int32_T *)&n_bs;
    g_bs.numDimensions = 1;
    c_bs.data[i] = g_bs.data[i] / ws->data[i];
  }

  i = bs->size[0];
  while (ncols + 1 <= i) {
    o_bs[0] = bs->size[0];
    o_bs[1] = 1;
    c_bs = *bs;
    c_bs.size = (int32_T *)&o_bs;
    c_bs.numDimensions = 1;
    c_bs.data[ncols] = 0.0;
    ncols++;
  }

  return deg_out;
}

static boolean_T compute_weights(const emxArray_real_T *us, const
  emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws)
{
  boolean_T toocoarse;
  boolean_T interp;
  int32_T vlen;
  int32_T j;
  real_T b[2];
  real_T y;
  int32_T iy;
  int32_T k;
  real_T h;
  real_T b_b[3];
  real_T costheta;
  real_T u1;

  /*  Compute weights for polynomial fitting. */
  /*  [ws,toocoarse] = compute_weights( us, nrms, deg, tol) */
  /*  */
  /*  Note that if size(us,1)==int32(size(nrms,1)) or size(us,1)==int32(size(nrms,1))-1. */
  /*  In the former, polyfit is approximate; and in the latter, */
  /*  polyfit is interpolatory. */
  interp = (nrms->size[0] - us->size[0] != 0);

  /*  First, compute squared distance from each input point to the pos */
  vlen = ws->size[0];
  ws->size[0] = us->size[0];
  emxEnsureCapacity((emxArray__common *)ws, vlen, (int32_T)sizeof(real_T));
  for (j = 0; j + 1 <= us->size[0]; j++) {
    for (vlen = 0; vlen < 2; vlen++) {
      b[vlen] = us->data[j + us->size[0] * vlen];
    }

    y = 0.0;
    vlen = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      y += us->data[j + us->size[0] * vlen] * b[iy];
      vlen++;
      iy++;
    }

    ws->data[j] = y;
  }

  /*  Second, compute a small correction term to guard aganst zero */
  if (ws->size[0] == 0) {
    y = 0.0;
  } else {
    vlen = ws->size[0];
    y = ws->data[0];
    for (k = 2; k <= vlen; k++) {
      y += ws->data[k - 1];
    }
  }

  h = y / (real_T)ws->size[0];

  /*  Finally, compute the weights for each vertex */
  toocoarse = FALSE;
  for (j = 0; j + 1 <= us->size[0]; j++) {
    for (vlen = 0; vlen < 3; vlen++) {
      b_b[vlen] = nrms->data[nrms->size[0] * vlen];
    }

    costheta = 0.0;
    vlen = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      costheta += nrms->data[(j + interp) + nrms->size[0] * vlen] * b_b[iy];
      vlen++;
      iy++;
    }

    if (costheta > 0.0) {
      y = ws->data[j] / h + 0.01;
      u1 = -(real_T)deg / 2.0;
      y = pow(y, u1);
      ws->data[j] = costheta * y;
    } else {
      ws->data[j] = 0.0;
      toocoarse = TRUE;
    }
  }

  return toocoarse;
}

static int32_T eval_vander_bivar_cmf(const emxArray_real_T *us, emxArray_real_T *
  bs, int32_T degree, const emxArray_real_T *ws)
{
  int32_T deg_out;
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T ii;
  int32_T nv;
  uint32_T jj;
  int32_T b_bs[2];
  int32_T c_bs[2];
  emxArray_real_T d_bs;
  emxArray_real_T e_bs;
  emxArray_real_T *ts;
  emxArray_real_T *b_V;
  emxArray_real_T *D;
  emxArray_real_T *v;
  int32_T rnk;
  int32_T nrows;
  int32_T k;
  boolean_T exitg2;
  real_T t2;
  real_T t;
  int32_T exitg1;
  int32_T f_bs[2];
  int32_T g_bs[2];
  int32_T h_bs[2];

  /* EVAL_VANDER_BIVAR_CMF Evaluate generalized Vandermonde matrix. */
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR_CMF(US,BS,DEGREE,WS, INTERP, SAFEGUARD) */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*  */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  Note: the only difference from EVAL_VANDER_UNIVAR is ws is not */
  /*        computed inside this function */
  /*  */
  /*  See also EVAL_VANDER_BIVAR */
  /*  Determine degree of fitting */
  npnts = us->size[0];

  /*  Declaring the degree of output */
  /*  Determine degree of polynomial */
  ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
  while ((npnts < ncols) && (degree > 1)) {
    degree--;
    ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
  }

  b_emxInit_real_T(&V, 2);

  /* % Construct matrix */
  gen_vander_bivar(us, degree, V);

  /* % Scale rows to assign different weights to different points */
  if (!(ws->size[0] == 0)) {
    for (ii = 0; ii + 1 <= npnts; ii++) {
      nv = V->size[1];
      for (jj = 1U; (real_T)jj <= (real_T)nv; jj++) {
        V->data[ii + V->size[0] * ((int32_T)jj - 1)] *= ws->data[ii];
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
      d_bs.data[ii] = e_bs.data[ii] * ws->data[ii];
    }
  }

  emxInit_real_T(&ts, 1);

  /* % Scale columns to reduce condition number */
  npnts = ts->size[0];
  ts->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ts, npnts, (int32_T)sizeof(real_T));

  /* % Rescale the columns of a matrix to reduce condition number */
  ii = 0;
  emxInit_real_T(&b_V, 1);
  while (ii + 1 <= ncols) {
    npnts = b_V->size[0];
    b_V->size[0] = V->size[0];
    emxEnsureCapacity((emxArray__common *)b_V, npnts, (int32_T)sizeof(real_T));
    nv = V->size[0] - 1;
    for (npnts = 0; npnts <= nv; npnts++) {
      b_V->data[npnts] = V->data[npnts + V->size[0] * ii];
    }

    ts->data[ii] = norm2_vec(b_V);
    if (fabs(ts->data[ii]) == 0.0) {
      ts->data[ii] = 1.0;
    } else {
      nv = V->size[0];
      for (npnts = 0; npnts + 1 <= nv; npnts++) {
        V->data[npnts + V->size[0] * ii] /= ts->data[ii];
      }
    }

    ii++;
  }

  emxFree_real_T(&b_V);
  emxInit_real_T(&D, 1);
  emxInit_real_T(&v, 1);

  /* % Perform Householder QR factorization */
  npnts = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, npnts, (int32_T)sizeof(real_T));

  /*  Compute Householder QR factorization with safeguards. */
  /*  It compares the diagonal entries with the given tolerance to */
  /*  determine whether the matrix is nearly singular. It is */
  /*  specialized for performing polynomial fittings. */
  /*  */
  /*  It saves Householder reflector vectors into lower triangular part A. */
  /*  Save diagonal part of R into D, and upper triangular part (excluding */
  /*  diagonal) of R into upper triangular part of A. */
  rnk = ncols;
  nrows = V->size[0];
  npnts = v->size[0];
  v->size[0] = nrows;
  emxEnsureCapacity((emxArray__common *)v, npnts, (int32_T)sizeof(real_T));
  k = 0;
  exitg2 = 0U;
  while ((exitg2 == 0U) && (k + 1 <= ncols)) {
    nv = nrows - k;
    for (npnts = 0; npnts + 1 <= nv; npnts++) {
      v->data[npnts] = V->data[(npnts + k) + V->size[0] * k];
    }

    /*  We don't need to worry about overflow, since A has been rescaled. */
    t2 = 0.0;
    for (npnts = 0; npnts + 1 <= nv; npnts++) {
      t2 += v->data[npnts] * v->data[npnts];
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
      for (npnts = 0; npnts + 1 <= nv; npnts++) {
        v->data[npnts] /= t2;
      }
    }

    /*  Optimized version for */
    /*  A(k:npnts,k:ncols) = A(k:npnts,k:ncols) - 2*v*(v'*A(k:npnts,k:ncols)); */
    for (npnts = k; npnts + 1 <= ncols; npnts++) {
      t2 = 0.0;
      for (ii = 0; ii + 1 <= nv; ii++) {
        t2 += v->data[ii] * V->data[(ii + k) + V->size[0] * npnts];
      }

      t2 += t2;
      for (ii = 0; ii + 1 <= nv; ii++) {
        V->data[(ii + k) + V->size[0] * npnts] -= t2 * v->data[ii];
      }
    }

    D->data[k] = V->data[k + V->size[0] * k];
    for (npnts = 0; npnts + 1 <= nv; npnts++) {
      V->data[(npnts + k) + V->size[0] * k] = v->data[npnts];
    }

    /*  Estimate rank of matrix */
    if (fabs(D->data[k]) < 1.0E-8) {
      rnk = k;
      exitg2 = 1U;
    } else {
      k++;
    }
  }

  emxFree_real_T(&v);

  /* % Adjust degree of fitting */
  do {
    exitg1 = 0U;
    if (rnk < ncols) {
      degree--;
      if (degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        npnts = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, npnts, (int32_T)sizeof(real_T));
        nv = bs->size[0] - 1;
        for (npnts = 0; npnts <= nv; npnts++) {
          bs->data[npnts] = 0.0;
        }

        exitg1 = 1U;
      } else {
        ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
      }
    } else {
      /* % Compute Q'bs */
      npnts = V->size[0];
      for (k = 0; k + 1 <= ncols; k++) {
        /*  Optimized version for */
        /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
        /*  where v is Q(k:npngs) */
        t2 = 0.0;
        for (ii = k; ii + 1 <= npnts; ii++) {
          f_bs[0] = bs->size[0];
          f_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&f_bs;
          d_bs.numDimensions = 1;
          t2 += V->data[ii + V->size[0] * k] * d_bs.data[ii];
        }

        t2 += t2;
        for (ii = k; ii + 1 <= npnts; ii++) {
          g_bs[0] = bs->size[0];
          g_bs[1] = 1;
          h_bs[0] = bs->size[0];
          h_bs[1] = 1;
          d_bs = *bs;
          d_bs.size = (int32_T *)&g_bs;
          d_bs.numDimensions = 1;
          e_bs = *bs;
          e_bs.size = (int32_T *)&h_bs;
          e_bs.numDimensions = 1;
          d_bs.data[ii] = e_bs.data[ii] - t2 * V->data[ii + V->size[0] * k];
        }
      }

      /* % Perform backward substitution and scale the solutions. */
      for (npnts = 0; npnts + 1 <= ncols; npnts++) {
        V->data[npnts + V->size[0] * npnts] = D->data[npnts];
      }

      deg_out = backsolve_bivar_safeguarded(V, bs, degree, ts);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ts);
  emxFree_real_T(&V);
  return deg_out;
}

static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V)
{
  int32_T npnts;
  emxArray_real_T *b_us;
  int32_T ncols;
  int32_T i4;
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
  npnts = us->size[0];
  if (degree <= 0) {
    emxInit_real_T(&b_us, 1);
    degree = -degree;
    ncols = (1 + degree) * (1 + degree);
    i4 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = ncols;
    emxEnsureCapacity((emxArray__common *)V, i4, (int32_T)sizeof(real_T));

    /*  Preallocate storage */
    /*  Use tensor product */
    i4 = b_us->size[0];
    b_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)b_us, i4, (int32_T)sizeof(real_T));
    c = us->size[0] - 1;
    for (i4 = 0; i4 <= c; i4++) {
      b_us->data[i4] = us->data[i4];
    }

    b_emxInit_real_T(&v1, 2);
    emxInit_real_T(&c_us, 1);
    gen_vander_univar(b_us, degree, v1);
    i4 = c_us->size[0];
    c_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)c_us, i4, (int32_T)sizeof(real_T));
    emxFree_real_T(&b_us);
    c = us->size[0] - 1;
    for (i4 = 0; i4 <= c; i4++) {
      c_us->data[i4] = us->data[i4 + us->size[0]];
    }

    b_emxInit_real_T(&v2, 2);
    gen_vander_univar(c_us, degree, v2);
    p = 0;
    emxFree_real_T(&c_us);
    b_emxInit_real_T(&r0, 2);
    b_emxInit_real_T(&y, 2);
    emxInit_real_T(&a, 1);
    b_emxInit_real_T(&b_v2, 2);
    while (p + 1 <= npnts) {
      i4 = a->size[0];
      a->size[0] = v1->size[1];
      emxEnsureCapacity((emxArray__common *)a, i4, (int32_T)sizeof(real_T));
      c = v1->size[1] - 1;
      for (i4 = 0; i4 <= c; i4++) {
        a->data[i4] = v1->data[p + v1->size[0] * i4];
      }

      i4 = b_v2->size[0] * b_v2->size[1];
      b_v2->size[0] = 1;
      b_v2->size[1] = v2->size[1];
      emxEnsureCapacity((emxArray__common *)b_v2, i4, (int32_T)sizeof(real_T));
      c = v2->size[1] - 1;
      for (i4 = 0; i4 <= c; i4++) {
        b_v2->data[b_v2->size[0] * i4] = v2->data[p + v2->size[0] * i4];
      }

      i4 = y->size[0] * y->size[1];
      y->size[0] = a->size[0];
      y->size[1] = b_v2->size[1];
      emxEnsureCapacity((emxArray__common *)y, i4, (int32_T)sizeof(real_T));
      c = b_v2->size[1] - 1;
      for (i4 = 0; i4 <= c; i4++) {
        nx = a->size[0] - 1;
        for (kk2 = 0; kk2 <= nx; kk2++) {
          y->data[kk2 + y->size[0] * i4] = a->data[kk2] * b_v2->data[b_v2->size
            [0] * i4];
        }
      }

      nx = y->size[0] * y->size[1];
      for (i4 = 0; i4 < 2; i4++) {
        sz[i4] = 0;
      }

      sz[0] = 1;
      sz[1] = ncols;
      i4 = r0->size[0] * r0->size[1];
      r0->size[0] = 1;
      r0->size[1] = sz[1];
      emxEnsureCapacity((emxArray__common *)r0, i4, (int32_T)sizeof(real_T));
      for (c = 0; c + 1 <= nx; c++) {
        r0->data[c] = y->data[c];
      }

      c = r0->size[1] - 1;
      for (i4 = 0; i4 <= c; i4++) {
        V->data[p + V->size[0] * i4] = r0->data[r0->size[0] * i4];
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
    i4 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = (int32_T)iv2[degree];
    emxEnsureCapacity((emxArray__common *)V, i4, (int32_T)sizeof(real_T));
    c = npnts * iv2[degree] - 1;
    for (i4 = 0; i4 <= c; i4++) {
      V->data[i4] = 0.0;
    }

    /*  Preallocate storage */
    /*     %% Compute rows corresponding to function values */
    if (1 > npnts) {
      i4 = 0;
    } else {
      i4 = npnts;
    }

    b_emxInit_int32_T(&r1, 1);
    kk2 = r1->size[0];
    r1->size[0] = i4;
    emxEnsureCapacity((emxArray__common *)r1, kk2, (int32_T)sizeof(int32_T));
    c = i4 - 1;
    for (i4 = 0; i4 <= c; i4++) {
      r1->data[i4] = 1 + i4;
    }

    c = r1->size[0];
    emxFree_int32_T(&r1);
    c--;
    for (i4 = 0; i4 <= c; i4++) {
      V->data[i4] = 1.0;
    }

    for (i4 = 0; i4 < 2; i4++) {
      iv3[i4] = (int8_T)(i4 + 1);
    }

    for (i4 = 0; i4 < 2; i4++) {
      c = us->size[0] - 1;
      for (kk2 = 0; kk2 <= c; kk2++) {
        V->data[kk2 + V->size[0] * iv3[i4]] = us->data[kk2 + us->size[0] * i4];
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

static int32_T mrdivide(int32_T A, real_T B)
{
  real_T d0;
  d0 = (real_T)A / B;
  if ((d0 < 4.503599627370496E+15) && (d0 > -4.503599627370496E+15)) {
    d0 = d0 < 0.0 ? ceil(d0 - 0.5) : floor(d0 + 0.5);
  }

  return (int32_T)d0;
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

static int32_T b_obtain_nring_surf(int32_T vid, int32_T ring, real_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, real_T ngbvs[128], emxArray_real_T *vtags,
  emxArray_real_T *ftags)
{
  int32_T nverts;
  int32_T fid;
  int32_T lid;
  int32_T nfaces;
  boolean_T overflow;
  boolean_T b0;
  int32_T fid_in;
  static const int8_T iv6[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv7[3] = { 3, 1, 2 };

  int32_T ngbfs[256];
  int32_T opp;
  int32_T nverts_pre;
  int32_T nfaces_pre;
  int32_T b_minpnts;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T b1;
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
    if ((ring == 1) && (minpnts == 0.0)) {
      b0 = TRUE;
    } else {
      b0 = FALSE;
    }

    /*  Optimized version for collecting one-ring vertices */
    if (opphes->data[fid + opphes->size[0] * lid] != 0) {
      fid_in = fid + 1;
    } else {
      fid_in = 0;
      nverts = 1;
      ngbvs[0] = (real_T)tris->data[fid + tris->size[0] * (iv6[lid] - 1)];
      if (!b0) {
        hebuf[0] = 0;
      }
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    do {
      exitg4 = 0U;

      /*  Insert vertx into list */
      lid = iv7[lid] - 1;
      if ((nverts < 128) && (nfaces < 256)) {
        nverts++;
        ngbvs[nverts - 1] = (real_T)tris->data[fid + tris->size[0] * lid];
        if (!b0) {
          /*  Save starting position for next vertex */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv7[lid] - 1)];
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
    if ((ring == 1) && (((real_T)nverts >= minpnts) || (nverts >= 128) ||
                        (nfaces >= 256))) {
    } else {
      vtags->data[vid - 1] = 1.0;
      for (lid = 1; lid <= nverts; lid++) {
        vtags->data[(int32_T)ngbvs[lid - 1] - 1] = 1.0;
      }

      for (lid = 1; lid <= nfaces; lid++) {
        ftags->data[ngbfs[lid - 1] - 1] = 1.0;
      }

      /*  Define buffers and prepare tags for further processing */
      nverts_pre = 0;
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      if (minpnts > 128.0) {
        b_minpnts = 128;
      } else {
        cur_ring = minpnts;
        if ((cur_ring < 4.503599627370496E+15) && (cur_ring >
             -4.503599627370496E+15)) {
          cur_ring = cur_ring < 0.0 ? ceil(cur_ring - 0.5) : floor(cur_ring +
            0.5);
        }

        b_minpnts = (int32_T)cur_ring;
      }

      cur_ring = 1.0;
      do {
        exitg1 = 0U;
        guard1 = FALSE;
        if (cur_ring > (real_T)ring) {
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
                   != 0) && (!(ftags->data[fid] != 0.0))) {
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);
                if (overflow || ((!(vtags->data[tris->data[fid + tris->size[0] *
                                    (iv7[lid] - 1)] - 1] != 0.0)) && (nverts >=
                      128)) || ((!(ftags->data[fid] != 0.0)) && (nfaces >= 256)))
                {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                if ((!(ftags->data[fid] != 0.0)) && (!overflow)) {
                  nfaces++;
                  ngbfs[nfaces - 1] = fid + 1;
                  ftags->data[fid] = 1.0;
                }

                if ((!(vtags->data[tris->data[fid + tris->size[0] * (iv7[lid] -
                       1)] - 1] != 0.0)) && (!overflow)) {
                  nverts++;
                  ngbvs[nverts - 1] = (real_T)tris->data[fid + tris->size[0] *
                    (iv7[lid] - 1)];
                  vtags->data[tris->data[fid + tris->size[0] * (iv7[lid] - 1)] -
                    1] = 1.0;
                }

                exitg2 = 1U;
              } else {
                lid++;
              }
            }

            nfaces_pre++;
          }

          if ((nverts >= b_minpnts) || (nverts >= 128) || (nfaces >= 256) ||
              (nfaces == opp)) {
            exitg1 = 1U;
          } else {
            /*  If needs to expand, then undo the last half ring */
            for (lid = nverts_last; lid + 1 <= nverts; lid++) {
              vtags->data[(int32_T)ngbvs[lid] - 1] = 0.0;
            }

            nverts = nverts_last;
            for (lid = opp; lid + 1 <= nfaces; lid++) {
              ftags->data[ngbfs[lid] - 1] = 0.0;
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
            fid = (int32_T)((uint32_T)v2he->data[(int32_T)ngbvs[nverts_pre] - 1]
                            >> 2U) - 1;

            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            lid = (int32_T)((uint32_T)v2he->data[(int32_T)ngbvs[nverts_pre] - 1]
                            & 3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            if ((hebuf[nverts_pre] != 0) && (opphes->data[fid + opphes->size[0] *
                 lid] != 0)) {
              b1 = TRUE;
            } else {
              b1 = FALSE;
            }

            if (b1) {
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
              if (overflow || ((!(vtags->data[tris->data[fid + tris->size[0] *
                                  (iv6[lid] - 1)] - 1] != 0.0)) && (nverts >=
                    128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              if (!overflow) {
                nverts++;
                ngbvs[nverts - 1] = (real_T)tris->data[fid + tris->size[0] *
                  (iv6[lid] - 1)];
                vtags->data[tris->data[fid + tris->size[0] * (iv6[lid] - 1)] - 1]
                  = 1.0;

                /*  Save starting position for next vertex */
                hebuf[nverts - 1] = 0;
              }
            }

            /*  Rotate counterclockwise around the vertex. */
            isfirst = TRUE;
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              lid = iv7[lid] - 1;

              /*  Insert face into list */
              guard2 = FALSE;
              if (ftags->data[fid] != 0.0) {
                if (b1 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                if (overflow || ((!(vtags->data[tris->data[fid + tris->size[0] *
                                    lid] - 1] != 0.0)) && (nverts >= 128)) || ((
                      !(ftags->data[fid] != 0.0)) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                if ((!(vtags->data[tris->data[fid + tris->size[0] * lid] - 1] !=
                       0.0)) && (!overflow)) {
                  nverts++;
                  ngbvs[nverts - 1] = (real_T)tris->data[fid + tris->size[0] *
                    lid];
                  vtags->data[tris->data[fid + tris->size[0] * lid] - 1] = 1.0;

                  /*  Save starting position for next ring */
                  hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv7[lid] - 1)];
                }

                if ((!(ftags->data[fid] != 0.0)) && (!overflow)) {
                  nfaces++;
                  ngbfs[nfaces - 1] = fid + 1;
                  ftags->data[fid] = 1.0;
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
          if (((nverts >= b_minpnts) && (cur_ring >= (real_T)ring)) || (nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      vtags->data[vid - 1] = 0.0;
      for (lid = 1; lid <= nverts; lid++) {
        vtags->data[(int32_T)ngbvs[lid - 1] - 1] = 0.0;
      }

      if (!b0) {
        for (lid = 1; lid <= nfaces; lid++) {
          ftags->data[ngbfs[lid - 1] - 1] = 0.0;
        }
      }
    }
  }

  return nverts;
}

static void polyfit3d_walf_tri(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, const emxArray_real_T *ngbpnts3, const emxArray_real_T *nrms3, const
  emxArray_real_T *xi, const emxArray_real_T *eta, int32_T deg, emxArray_real_T *
  pnt)
{
  emxArray_real_T *pos;
  int32_T np;
  int32_T i3;
  int32_T i;
  real_T d1;
  real_T b_xi;
  real_T b_eta;
  emxArray_real_T *pnt1;
  emxArray_real_T *pnt2;
  emxArray_real_T *b_pos;

  /*  Compute the position of a point within a triangle using */
  /*             weighted averaging of least-squares fittings. */
  /*  */
  /*  Input: */
  /*  ngbpnts1-3:Input points of size mx3, Its first column is x-coordinates, */
  /*             and its second column is y-coordinates. The first vertex will */
  /*             be used as the origin of the local coordinate system. */
  /*  nrms1-3:   The normals at ngbptns */
  /*  xi,eta:    The two parameters in the tangent plane */
  /*  deg:       The degree of polynomial to fit, from 1 to 6 */
  /*  interp:    If true, the fit is interpolatory at vertices. */
  /*  */
  /*  Output: */
  /*  pnt:       The reconstructed point in the global coordinate system */
  /*  */
  /*  See also polyfit3d_walf_quad, polyfit3d_walf_edge, polyfit3d_cmf_tri */
  /*  Use quadratic fitting by default */
  if (deg == 0) {
    deg = 2;
  }

  b_emxInit_real_T(&pos, 2);

  /*  Do not use interpolation by default */
  /*  Compute face normal vector and the local coordinate */
  np = xi->size[0];
  i3 = pos->size[0] * pos->size[1];
  pos->size[0] = np;
  pos->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)pos, i3, (int32_T)sizeof(real_T));
  for (i = 0; i + 1 <= np; i++) {
    d1 = (1.0 - xi->data[i]) - eta->data[i];
    b_xi = xi->data[i];
    b_eta = eta->data[i];
    for (i3 = 0; i3 < 3; i3++) {
      pos->data[i + pos->size[0] * i3] = (d1 * ngbpnts1->data[ngbpnts1->size[0] *
        i3] + b_xi * ngbpnts2->data[ngbpnts2->size[0] * i3]) + b_eta *
        ngbpnts3->data[ngbpnts3->size[0] * i3];
    }
  }

  b_emxInit_real_T(&pnt1, 2);
  b_emxInit_real_T(&pnt2, 2);
  b_emxInit_real_T(&b_pos, 2);

  /* pos = (1-xi-eta).*ngbpnts1(1,1:3) +xi*ngbpnts2(1,1:3)+eta*ngbpnts3(1,1:3); */
  /*  Interpolate using vertex-based polynomial fittings at three vertices */
  polyfit3d_walf_vertex(ngbpnts1, nrms1, pos, deg, pnt1);
  polyfit3d_walf_vertex(ngbpnts2, nrms2, pos, deg, pnt2);
  i3 = b_pos->size[0] * b_pos->size[1];
  b_pos->size[0] = pos->size[0];
  b_pos->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)b_pos, i3, (int32_T)sizeof(real_T));
  i = pos->size[0] * pos->size[1] - 1;
  for (i3 = 0; i3 <= i; i3++) {
    b_pos->data[i3] = pos->data[i3];
  }

  polyfit3d_walf_vertex(ngbpnts3, nrms3, b_pos, deg, pos);

  /*  Compute weighted average of the three points */
  /* pnt = (1-xi-eta).*pnt1 + xi.*pnt2 + eta.*pnt3; */
  i3 = pnt->size[0] * pnt->size[1];
  pnt->size[0] = np;
  pnt->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)pnt, i3, (int32_T)sizeof(real_T));
  i = 0;
  emxFree_real_T(&b_pos);
  while (i + 1 <= np) {
    d1 = (1.0 - xi->data[i]) - eta->data[i];
    b_xi = xi->data[i];
    b_eta = eta->data[i];
    for (i3 = 0; i3 < 3; i3++) {
      pnt->data[i + pnt->size[0] * i3] = (d1 * pnt1->data[i + pnt1->size[0] * i3]
        + b_xi * pnt2->data[i + pnt2->size[0] * i3]) + b_eta * pos->data[i +
        pos->size[0] * i3];
    }

    i++;
  }

  emxFree_real_T(&pnt2);
  emxFree_real_T(&pnt1);
  emxFree_real_T(&pos);
}

static void polyfit3d_walf_vertex(const emxArray_real_T *pnts, const
  emxArray_real_T *nrms, const emxArray_real_T *pos, int32_T deg,
  emxArray_real_T *pnt)
{
  int32_T iy;
  real_T nrm[3];
  real_T absnrm[3];
  int32_T i;
  static const int8_T iv0[3] = { 0, 1, 0 };

  static const int8_T iv1[3] = { 1, 0, 0 };

  real_T u;
  int32_T ix;
  int32_T k;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  real_T t2[3];
  int32_T nverts;
  int32_T b_ix;
  real_T vec[3];
  emxArray_real_T *ws_row;
  emxArray_real_T *b_ws_row;
  boolean_T toocoarse;
  int32_T b_iy;
  emxArray_real_T *us1;
  emxArray_real_T *ws_row1;
  int32_T c_ws_row[2];
  int32_T b_bs[2];
  emxArray_real_T d_ws_row;
  emxArray_real_T c_bs;
  int32_T e_ws_row[2];
  emxArray_real_T *d_bs;
  int32_T np;
  real_T v;
  real_T V[28];
  real_T height;

  /*  Construct a local polynomial fitting and then interpolate. */
  /*     The function constructs the fitting using points pnts(:,1:3) in */
  /*     a local uv coordinate system with pnts(1,1:3) as the origin */
  /*     and nrms(1,1:3) vertical axis, and then interpolates to the point */
  /*     with u=param to obtain its coordinates. */
  /*  */
  /*  Input: */
  /*  pnts:   Input points of size mx3, Its first column is x-coordinates, */
  /*          and its second column is y-coordinates. The first vertex will */
  /*          be used as the origin of the local coordinate system. */
  /*  nrms:   The normals at pnts */
  /*  pos:    The point to be interpolated. */
  /*  deg:    The degree of polynomial to fit, from 0 to 6 */
  /*  interp:  If 1, the fit is interpolatory at pnts(1,:) (in other words, */
  /*          the fit passes through point pnts(1,:)). If 0, the fit does not */
  /*          pass the point pnts(1,:), useful for a noisy inputs. */
  /*  */
  /*  Output: */
  /*  pnt:    The interpolated coordinates in the global coordinate system */
  /*  */
  for (iy = 0; iy < 3; iy++) {
    nrm[iy] = nrms->data[nrms->size[0] * iy];
  }

  b_abs(nrm, absnrm);
  if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
    for (i = 0; i < 3; i++) {
      absnrm[i] = (real_T)iv0[i];
    }
  } else {
    for (i = 0; i < 3; i++) {
      absnrm[i] = (real_T)iv1[i];
    }
  }

  u = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    u += absnrm[ix] * nrm[iy];
    ix++;
    iy++;
  }

  for (iy = 0; iy < 3; iy++) {
    absnrm[iy] -= u * nrm[iy];
  }

  u = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    u += absnrm[ix] * absnrm[iy];
    ix++;
    iy++;
  }

  u = sqrt(u);
  for (iy = 0; iy < 3; iy++) {
    absnrm[iy] /= u;
  }

  b_emxInit_real_T(&us, 2);
  emxInit_real_T(&bs, 1);

  /* CROSS_COL Efficient routine for computing cross product of two  */
  /* 3-dimensional column vectors. */
  /*  CROSS_COL(A,B) Efficiently computes the cross product between */
  /*  3-dimensional column vector A, and 3-dimensional column vector B. */
  t2[0] = nrm[1] * absnrm[2] - nrm[2] * absnrm[1];
  t2[1] = nrm[2] * absnrm[0] - nrm[0] * absnrm[2];
  t2[2] = nrm[0] * absnrm[1] - nrm[1] * absnrm[0];

  /* % Project onto local coordinate system */
  nverts = pnts->size[0];
  iy = us->size[0] * us->size[1];
  us->size[0] = nverts;
  us->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)us, iy, (int32_T)sizeof(real_T));
  iy = bs->size[0];
  bs->size[0] = nverts;
  emxEnsureCapacity((emxArray__common *)bs, iy, (int32_T)sizeof(real_T));
  for (iy = 0; iy < 2; iy++) {
    us->data[us->size[0] * iy] = 0.0;
  }

  for (b_ix = 0; b_ix + 1 <= nverts; b_ix++) {
    for (iy = 0; iy < 3; iy++) {
      vec[iy] = pnts->data[b_ix + pnts->size[0] * iy] - pnts->data[pnts->size[0]
        * iy];
    }

    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += vec[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[b_ix] = u;
    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += vec[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[b_ix + us->size[0]] = u;
    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += vec[ix] * nrm[iy];
      ix++;
      iy++;
    }

    bs->data[b_ix] = u;
  }

  emxInit_real_T(&ws_row, 1);
  emxInit_real_T(&b_ws_row, 1);
  toocoarse = compute_weights(us, nrms, deg, b_ws_row);
  iy = ws_row->size[0];
  ws_row->size[0] = b_ws_row->size[0];
  emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
  b_iy = b_ws_row->size[0] - 1;
  for (iy = 0; iy <= b_iy; iy++) {
    ws_row->data[iy] = b_ws_row->data[iy];
  }

  /*  Compute the coefficients and store into bs */
  if (toocoarse) {
    b_emxInit_real_T(&us1, 2);
    emxInit_real_T(&ws_row1, 1);
    iy = us1->size[0] * us1->size[1];
    us1->size[0] = nverts;
    us1->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us1, iy, (int32_T)sizeof(real_T));
    iy = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
    iy = ws_row1->size[0];
    ws_row1->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row1, iy, (int32_T)sizeof(real_T));
    nverts = 0;
    for (i = 0; i + 1 <= us->size[0]; i++) {
      if (b_ws_row->data[i] > 0.0) {
        nverts++;
        for (iy = 0; iy < 2; iy++) {
          us1->data[(nverts + us1->size[0] * iy) - 1] = us->data[i + us->size[0]
            * iy];
        }

        c_ws_row[0] = ws_row->size[0];
        c_ws_row[1] = 1;
        b_bs[0] = bs->size[0];
        b_bs[1] = 1;
        d_ws_row = *ws_row;
        d_ws_row.size = (int32_T *)&c_ws_row;
        d_ws_row.numDimensions = 1;
        c_bs = *bs;
        c_bs.size = (int32_T *)&b_bs;
        c_bs.numDimensions = 1;
        d_ws_row.data[nverts - 1] = c_bs.data[i];
        ws_row1->data[nverts - 1] = b_ws_row->data[i];
      }
    }

    if (1 > nverts) {
      iy = 0;
    } else {
      iy = nverts;
    }

    b_ix = us->size[0] * us->size[1];
    us->size[0] = iy;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, b_ix, (int32_T)sizeof(real_T));
    for (b_ix = 0; b_ix < 2; b_ix++) {
      b_iy = iy - 1;
      for (ix = 0; ix <= b_iy; ix++) {
        us->data[ix + us->size[0] * b_ix] = us1->data[ix + us1->size[0] * b_ix];
      }
    }

    emxFree_real_T(&us1);
    if (1 > nverts) {
      iy = 0;
    } else {
      iy = nverts;
    }

    e_ws_row[0] = ws_row->size[0];
    e_ws_row[1] = 1;
    b_ix = bs->size[0];
    bs->size[0] = iy;
    emxEnsureCapacity((emxArray__common *)bs, b_ix, (int32_T)sizeof(real_T));
    b_iy = iy - 1;
    for (iy = 0; iy <= b_iy; iy++) {
      d_ws_row = *ws_row;
      d_ws_row.size = (int32_T *)&e_ws_row;
      d_ws_row.numDimensions = 1;
      bs->data[iy] = d_ws_row.data[iy];
    }

    if (1 > nverts) {
      nverts = 0;
    }

    iy = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
    b_iy = nverts - 1;
    for (iy = 0; iy <= b_iy; iy++) {
      ws_row->data[iy] = ws_row1->data[iy];
    }

    emxFree_real_T(&ws_row1);
  }

  emxFree_real_T(&b_ws_row);
  emxInit_real_T(&d_bs, 1);
  iy = d_bs->size[0];
  d_bs->size[0] = bs->size[0];
  emxEnsureCapacity((emxArray__common *)d_bs, iy, (int32_T)sizeof(real_T));
  b_iy = bs->size[0] - 1;
  for (iy = 0; iy <= b_iy; iy++) {
    d_bs->data[iy] = bs->data[iy];
  }

  emxFree_real_T(&bs);
  nverts = eval_vander_bivar_cmf(us, d_bs, deg, ws_row);

  /* % project the point into u-v plane and evaluate its value */
  /*  vec = (pos - pnts(1,1:3)).'; */
  /*   */
  /*  u = vec.' * t1; */
  /*  v = vec.' * t2; */
  /*   */
  /*  % Evaluate the polynomial */
  /*  V = coder.nullcopy(zeros(28,1)); */
  /*  V(1) = u; V(2) = v; */
  /*  jj = int32(2); */
  /*   */
  /*  for kk=2:deg_out */
  /*      jj = jj + 1; V(jj) = V(jj-kk)*u; */
  /*   */
  /*      for kk2=1:kk */
  /*          jj = jj + 1; V(jj) = V(jj-kk-1)*v; */
  /*      end */
  /*  end */
  /*   */
  /*  if interp; height = 0; else height = bs(1); end */
  /*  for kk=1:jj */
  /*      height = height + bs(kk+1-int32(interp)) * V(kk); */
  /*  end */
  /*   */
  /*   */
  /*  %% Change back to global coordinate system. */
  /*  pnt = pnts(1,1:3)' + u*t1 + v*t2 + height*nrm; */
  /* % */
  np = pos->size[0];
  iy = pnt->size[0] * pnt->size[1];
  pnt->size[0] = np;
  pnt->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)pnt, iy, (int32_T)sizeof(real_T));
  i = 0;
  emxFree_real_T(&ws_row);
  emxFree_real_T(&us);
  while (i + 1 <= np) {
    for (iy = 0; iy < 3; iy++) {
      vec[iy] = pos->data[i + pos->size[0] * iy] - pnts->data[pnts->size[0] * iy];
    }

    u = 0.0;
    ix = 0;
    iy = 0;
    v = 0.0;
    b_ix = 0;
    b_iy = 0;
    for (k = 0; k < 3; k++) {
      u += vec[ix] * absnrm[iy];
      ix++;
      iy++;
      v += vec[b_ix] * t2[b_iy];
      b_ix++;
      b_iy++;
    }

    /* Evaluating the polynomial for height function */
    V[0] = u;
    V[1] = v;
    ix = 1;
    for (b_ix = 2; b_ix <= nverts; b_ix++) {
      ix++;
      V[ix] = V[ix - b_ix] * u;
      for (b_iy = 1; b_iy <= b_ix; b_iy++) {
        ix++;
        V[ix] = V[(ix - b_ix) - 1] * v;
      }
    }

    height = d_bs->data[0];
    for (b_ix = 1; b_ix <= ix + 1; b_ix++) {
      height += d_bs->data[b_ix] * V[b_ix - 1];
    }

    for (iy = 0; iy < 3; iy++) {
      pnt->data[i + pnt->size[0] * iy] = ((pnts->data[pnts->size[0] * iy] + u *
        absnrm[iy]) + v * t2[iy]) + height * nrm[iy];
    }

    i++;
  }

  emxFree_real_T(&d_bs);
}

void test_walf_tri(const emxArray_real_T *ps, const emxArray_int32_T *tris,
                   int32_T degree, const emxArray_real_T *param, emxArray_real_T
                   *pnts)
{
  int32_T nv;
  int32_T i0;
  uint32_T uv0[2];
  emxArray_int32_T *opphes;
  int32_T i;
  emxArray_int32_T *v2he;
  emxArray_real_T *vtags;
  emxArray_real_T *ftags;
  emxArray_real_T *nrms;
  int32_T ring;
  real_T y;
  uint32_T b_i;
  emxArray_real_T *ngbpnts1;
  emxArray_real_T *ngbpnts2;
  emxArray_real_T *ngbpnts3;
  emxArray_real_T *nrms1;
  emxArray_real_T *nrms2;
  emxArray_real_T *nrms3;
  emxArray_real_T *b_param;
  emxArray_real_T *c_param;
  emxArray_real_T *b_ngbpnts1;
  real_T ngbvs1[128];
  int32_T nverts1;
  real_T ngbvs2[128];
  int32_T nverts2;
  real_T ngbvs3[128];
  int32_T nverts3;
  int32_T i1;
  uint32_T j;

  /*  This function computes points projected onto the high order surface */
  /*  reconstructed using WALF. The inputs are  */
  /*  ps: points */
  /*  tris: triangles */
  /*  degree: degree of polynomial fitting  */
  /*  param: barycentric coordinates of points to be projected */
  /*  Output: pnts: the projected points , [numtris. 3*size(param,1)] */
  nv = ps->size[0];

  /*  Determine opposite halfedges */
  for (i0 = 0; i0 < 2; i0++) {
    uv0[i0] = (uint32_T)tris->size[i0];
  }

  emxInit_int32_T(&opphes, 2);
  i0 = opphes->size[0] * opphes->size[1];
  opphes->size[0] = (int32_T)uv0[0];
  opphes->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)opphes, i0, (int32_T)sizeof(int32_T));
  i = (int32_T)uv0[0] * 3 - 1;
  for (i0 = 0; i0 <= i; i0++) {
    opphes->data[i0] = 0;
  }

  b_emxInit_int32_T(&v2he, 1);
  determine_opposite_halfedge_tri((real_T)nv, tris, opphes);

  /*  Determine incident halfedge. */
  i0 = v2he->size[0];
  v2he->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)v2he, i0, (int32_T)sizeof(int32_T));
  i = nv - 1;
  for (i0 = 0; i0 <= i; i0++) {
    v2he->data[i0] = 0;
  }

  emxInit_real_T(&vtags, 1);
  determine_incident_halfedges(tris, opphes, v2he);
  i0 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i0, (int32_T)sizeof(real_T));
  i = nv - 1;
  for (i0 = 0; i0 <= i; i0++) {
    vtags->data[i0] = 0.0;
  }

  emxInit_real_T(&ftags, 1);
  i0 = ftags->size[0];
  ftags->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i0, (int32_T)sizeof(real_T));
  i = tris->size[0] - 1;
  for (i0 = 0; i0 <= i; i0++) {
    ftags->data[i0] = 0.0;
  }

  b_emxInit_real_T(&nrms, 2);
  average_vertex_normal_tri(ps, tris, nrms);
  ring = mrdivide(degree + 2, 2.0);
  y = 3.0 * (real_T)param->size[0];
  nv = tris->size[0];
  i0 = pnts->size[0] * pnts->size[1];
  pnts->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)pnts, i0, (int32_T)sizeof(real_T));
  i0 = pnts->size[0] * pnts->size[1];
  pnts->size[1] = (int32_T)y;
  emxEnsureCapacity((emxArray__common *)pnts, i0, (int32_T)sizeof(real_T));
  i = tris->size[0] * (int32_T)y - 1;
  for (i0 = 0; i0 <= i; i0++) {
    pnts->data[i0] = 0.0;
  }

  b_i = 1U;
  b_emxInit_real_T(&ngbpnts1, 2);
  b_emxInit_real_T(&ngbpnts2, 2);
  b_emxInit_real_T(&ngbpnts3, 2);
  b_emxInit_real_T(&nrms1, 2);
  b_emxInit_real_T(&nrms2, 2);
  b_emxInit_real_T(&nrms3, 2);
  emxInit_real_T(&b_param, 1);
  emxInit_real_T(&c_param, 1);
  b_emxInit_real_T(&b_ngbpnts1, 2);
  while (b_i <= (uint32_T)tris->size[0]) {
    for (i = 0; i < 128; i++) {
      ngbvs1[i] = 0.0;
    }

    nverts1 = b_obtain_nring_surf(tris->data[(int32_T)b_i - 1], ring, 5.0, tris,
      opphes, v2he, ngbvs1, vtags, ftags) + 1;
    for (i = 0; i < 128; i++) {
      ngbvs2[i] = 0.0;
    }

    nverts2 = b_obtain_nring_surf(tris->data[((int32_T)b_i + tris->size[0]) - 1],
      ring, 5.0, tris, opphes, v2he, ngbvs2, vtags, ftags) + 1;
    for (i = 0; i < 128; i++) {
      ngbvs3[i] = 0.0;
    }

    nverts3 = b_obtain_nring_surf(tris->data[((int32_T)b_i + (tris->size[0] << 1))
      - 1], ring, 5.0, tris, opphes, v2he, ngbvs3, vtags, ftags) + 1;
    i0 = ngbpnts1->size[0] * ngbpnts1->size[1];
    ngbpnts1->size[0] = nverts1;
    ngbpnts1->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)ngbpnts1, i0, (int32_T)sizeof(real_T));
    i = nverts1 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      ngbpnts1->data[i0] = 0.0;
    }

    i0 = ngbpnts2->size[0] * ngbpnts2->size[1];
    ngbpnts2->size[0] = nverts2;
    ngbpnts2->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)ngbpnts2, i0, (int32_T)sizeof(real_T));
    i = nverts2 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      ngbpnts2->data[i0] = 0.0;
    }

    i0 = ngbpnts3->size[0] * ngbpnts3->size[1];
    ngbpnts3->size[0] = nverts3;
    ngbpnts3->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)ngbpnts3, i0, (int32_T)sizeof(real_T));
    i = nverts3 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      ngbpnts3->data[i0] = 0.0;
    }

    nv = tris->data[(int32_T)b_i - 1];
    for (i0 = 0; i0 < 3; i0++) {
      ngbpnts1->data[ngbpnts1->size[0] * i0] = ps->data[(nv + ps->size[0] * i0)
        - 1];
    }

    if (1 > nverts1 - 1) {
      i0 = -1;
    } else {
      i0 = nverts1 - 2;
    }

    if (2 > nverts1) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        ngbpnts1->data[(nv + i1) + ngbpnts1->size[0] * i] = ps->data[((int32_T)
          ngbvs1[i1] + ps->size[0] * i) - 1];
      }
    }

    nv = tris->data[((int32_T)b_i + tris->size[0]) - 1];
    for (i0 = 0; i0 < 3; i0++) {
      ngbpnts2->data[ngbpnts2->size[0] * i0] = ps->data[(nv + ps->size[0] * i0)
        - 1];
    }

    if (1 > nverts2 - 1) {
      i0 = -1;
    } else {
      i0 = nverts2 - 2;
    }

    if (2 > nverts2) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        ngbpnts2->data[(nv + i1) + ngbpnts2->size[0] * i] = ps->data[((int32_T)
          ngbvs2[i1] + ps->size[0] * i) - 1];
      }
    }

    nv = tris->data[((int32_T)b_i + (tris->size[0] << 1)) - 1];
    for (i0 = 0; i0 < 3; i0++) {
      ngbpnts3->data[ngbpnts3->size[0] * i0] = ps->data[(nv + ps->size[0] * i0)
        - 1];
    }

    if (1 > nverts3 - 1) {
      i0 = -1;
    } else {
      i0 = nverts3 - 2;
    }

    if (2 > nverts3) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        ngbpnts3->data[(nv + i1) + ngbpnts3->size[0] * i] = ps->data[((int32_T)
          ngbvs3[i1] + ps->size[0] * i) - 1];
      }
    }

    i0 = nrms1->size[0] * nrms1->size[1];
    nrms1->size[0] = nverts1;
    nrms1->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)nrms1, i0, (int32_T)sizeof(real_T));
    i = nverts1 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      nrms1->data[i0] = 0.0;
    }

    i0 = nrms2->size[0] * nrms2->size[1];
    nrms2->size[0] = nverts2;
    nrms2->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)nrms2, i0, (int32_T)sizeof(real_T));
    i = nverts2 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      nrms2->data[i0] = 0.0;
    }

    i0 = nrms3->size[0] * nrms3->size[1];
    nrms3->size[0] = nverts3;
    nrms3->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)nrms3, i0, (int32_T)sizeof(real_T));
    i = nverts3 * 3 - 1;
    for (i0 = 0; i0 <= i; i0++) {
      nrms3->data[i0] = 0.0;
    }

    nv = tris->data[(int32_T)b_i - 1];
    for (i0 = 0; i0 < 3; i0++) {
      nrms1->data[nrms1->size[0] * i0] = nrms->data[(nv + nrms->size[0] * i0) -
        1];
    }

    if (1 > nverts1 - 1) {
      i0 = -1;
    } else {
      i0 = nverts1 - 2;
    }

    if (2 > nverts1) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        nrms1->data[(nv + i1) + nrms1->size[0] * i] = nrms->data[((int32_T)
          ngbvs1[i1] + nrms->size[0] * i) - 1];
      }
    }

    nv = tris->data[((int32_T)b_i + tris->size[0]) - 1];
    for (i0 = 0; i0 < 3; i0++) {
      nrms2->data[nrms2->size[0] * i0] = nrms->data[(nv + nrms->size[0] * i0) -
        1];
    }

    if (1 > nverts2 - 1) {
      i0 = -1;
    } else {
      i0 = nverts2 - 2;
    }

    if (2 > nverts2) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        nrms2->data[(nv + i1) + nrms2->size[0] * i] = nrms->data[((int32_T)
          ngbvs2[i1] + nrms->size[0] * i) - 1];
      }
    }

    nv = tris->data[((int32_T)b_i + (tris->size[0] << 1)) - 1];
    for (i0 = 0; i0 < 3; i0++) {
      nrms3->data[nrms3->size[0] * i0] = nrms->data[(nv + nrms->size[0] * i0) -
        1];
    }

    if (1 > nverts3 - 1) {
      i0 = -1;
    } else {
      i0 = nverts3 - 2;
    }

    if (2 > nverts3) {
      nv = 0;
    } else {
      nv = 1;
    }

    for (i = 0; i < 3; i++) {
      for (i1 = 0; i1 <= i0; i1++) {
        nrms3->data[(nv + i1) + nrms3->size[0] * i] = nrms->data[((int32_T)
          ngbvs3[i1] + nrms->size[0] * i) - 1];
      }
    }

    i0 = b_param->size[0];
    b_param->size[0] = param->size[0];
    emxEnsureCapacity((emxArray__common *)b_param, i0, (int32_T)sizeof(real_T));
    i = param->size[0] - 1;
    for (i0 = 0; i0 <= i; i0++) {
      b_param->data[i0] = param->data[i0];
    }

    i0 = c_param->size[0];
    c_param->size[0] = param->size[0];
    emxEnsureCapacity((emxArray__common *)c_param, i0, (int32_T)sizeof(real_T));
    i = param->size[0] - 1;
    for (i0 = 0; i0 <= i; i0++) {
      c_param->data[i0] = param->data[i0 + param->size[0]];
    }

    i0 = b_ngbpnts1->size[0] * b_ngbpnts1->size[1];
    b_ngbpnts1->size[0] = ngbpnts1->size[0];
    b_ngbpnts1->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)b_ngbpnts1, i0, (int32_T)sizeof(real_T));
    i = ngbpnts1->size[0] * ngbpnts1->size[1] - 1;
    for (i0 = 0; i0 <= i; i0++) {
      b_ngbpnts1->data[i0] = ngbpnts1->data[i0];
    }

    polyfit3d_walf_tri(b_ngbpnts1, nrms1, ngbpnts2, nrms2, ngbpnts3, nrms3,
                       b_param, c_param, degree, ngbpnts1);
    for (j = 1U; (real_T)j <= (real_T)ngbpnts1->size[0]; j++) {
      y = 3.0 * (real_T)j;
      i = (int32_T)b_i;
      nv = (int32_T)j;
      for (i0 = 0; i0 < 3; i0++) {
        pnts->data[(i + pnts->size[0] * ((int32_T)(y + (-2.0 + (real_T)i0)) - 1))
          - 1] = ngbpnts1->data[(nv + ngbpnts1->size[0] * i0) - 1];
      }
    }

    b_i++;
  }

  emxFree_real_T(&b_ngbpnts1);
  emxFree_real_T(&c_param);
  emxFree_real_T(&b_param);
  emxFree_real_T(&nrms3);
  emxFree_real_T(&nrms2);
  emxFree_real_T(&nrms1);
  emxFree_real_T(&ngbpnts3);
  emxFree_real_T(&ngbpnts2);
  emxFree_real_T(&ngbpnts1);
  emxFree_real_T(&nrms);
  emxFree_real_T(&ftags);
  emxFree_real_T(&vtags);
  emxFree_int32_T(&v2he);
  emxFree_int32_T(&opphes);
}

