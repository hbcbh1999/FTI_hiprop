#include "util.h"

static void accumulate_isometry_energy_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *elem_energies, emxArray_real_T
  *grads_smooth, emxArray_real_T *Hs_smooth);
static boolean_T add_disps_to_pntpositions(int32_T nv_clean, int32_T nt_clean,
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *us_smooth, real_T min_angle_pre, boolean_T scaled);
static boolean_T any(const boolean_T x[3]);
static boolean_T async_scale_disps_tri_cleanmesh(int32_T nv_clean, const
  emxArray_real_T *xs, emxArray_real_T *us_smooth, const emxArray_int32_T *tris);
static void b_abs(const real_T x[3], real_T y[3]);
static void b_backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
  cend, const emxArray_real_T *ws);
static void b_compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols);
static void b_determine_incident_halfedges(const emxArray_int32_T *elems, const
  emxArray_int32_T *opphes, emxArray_int32_T *v2he);
static void b_eigenanalysis_surf(const emxArray_real_T *As, const
  emxArray_real_T *bs, const emxArray_boolean_T *isridge, emxArray_real_T *us,
  emxArray_real_T *Vs);
static boolean_T b_eml_strcmp(const char_T a[3]);
static void b_emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int32_T
  numDimensions);
static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void b_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static void b_eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3]);
static int32_T b_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T
  *bs, const emxArray_real_T *ws);
static int32_T b_eval_vander_bivar_cmf(const emxArray_real_T *us,
  emxArray_real_T *bs, int32_T degree, const emxArray_real_T *ws);
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
static void b_rescale_displacements(const emxArray_real_T *xs, const
  emxArray_real_T *us, const emxArray_int32_T *tris, emxArray_real_T *alpha_vs);
static void b_sum(const real_T x[9], real_T y[3]);
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
                      cend, const emxArray_real_T *ws);
static int32_T backsolve_bivar_safeguarded(const emxArray_real_T *R,
  emxArray_real_T *bs, int32_T degree, const emxArray_real_T *ws);
static void c_adjust_disps_onto_hisurf_clea(int32_T nv_clean, const
  emxArray_real_T *ps, emxArray_real_T *us_smooth, const emxArray_real_T *nrms,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const char_T
  args_method[3], int32_T args_degree);
static void c_average_vertex_normal_tri_cle(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *flabel, emxArray_real_T *nrms);
static void c_compute_statistics_tris_clean(int32_T nt_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, real_T *min_angle, real_T
  *max_angle, real_T *min_area, real_T *max_area);
static void c_constrained_smooth_surf_clean(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_int32_T
  *flabels, const emxArray_boolean_T *isridge, const emxArray_real_T
  *grads_smooth, const emxArray_real_T *Hs_smooth, boolean_T check_trank,
  emxArray_real_T *us_smooth);
static boolean_T c_eml_strcmp(const char_T a[3]);
static void c_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void c_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);
static void c_eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws);
static void c_polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor,
  int32_T degree, real_T nrm[3], int32_T *deg, real_T prcurvs[2], real_T
  maxprdir[3]);
static void c_weighted_Laplacian_tri_cleanm(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_boolean_T
  *isridge, const emxArray_boolean_T *ridgeedge, const emxArray_int32_T *flabel,
  boolean_T check_trank, emxArray_real_T *us_smooth);
static real_T check_prism(const real_T xs[9], const real_T us[9]);
static boolean_T compute_cmf_weights(const real_T pos[3], const emxArray_real_T *
  pnts, const emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws);

static void b_compute_diffops_surf_cleanmesh(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *nrms_proj, int32_T degree, real_T ring, emxArray_real_T *nrms, const
  emxArray_real_T *curs, const emxArray_real_T *prdirs);

static void compute_hisurf_normals(int32_T nv_clean, const emxArray_real_T *xs,
  const emxArray_int32_T *tris, int32_T degree, emxArray_real_T *nrms);
static void compute_medial_quadric_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, const emxArray_int32_T *flabel, emxArray_real_T *As,
  emxArray_real_T *bs, emxArray_real_T *bs_lbl);
static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols);
static void compute_statistics_tris_global(int32_T nt_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, real_T *min_angle, real_T
  *max_angle, real_T *min_area, real_T *max_area);
static boolean_T compute_weights(const emxArray_real_T *us, const
  emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws);
static real_T cos_angle(const real_T ts1[3], const real_T ts2[3]);
static boolean_T d_eml_strcmp(const char_T a[3]);
static void d_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions);

static void c_determine_incident_halfedges(const emxArray_int32_T *elems, const
  emxArray_int32_T *opphes, emxArray_int32_T *v2he);
static void determine_opposite_halfedge(int32_T nv, const emxArray_int32_T
  *elems, emxArray_int32_T *opphes);


static boolean_T e_eml_strcmp(const char_T a[3]);
static void eig3_sorted(const real_T A[9], real_T V[9], real_T lambdas[3]);
static void eigenanalysis_surf(const emxArray_real_T *As, const emxArray_real_T *
  bs, const emxArray_boolean_T *isridge, emxArray_real_T *us, emxArray_real_T
  *Vs, emxArray_int8_T *tranks);
static boolean_T eml_strcmp(const char_T a[3]);


static void eval_curvature_lhf_surf(const real_T grad[2], const real_T H[4],
  real_T curvs[2], real_T dir[3]);
static void eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws);
static int32_T eval_vander_bivar_cmf(const emxArray_real_T *us, emxArray_real_T *
  bs, int32_T degree, const emxArray_real_T *ws);
static boolean_T f_eml_strcmp(const char_T a[3]);
static void find_parent_triangle(const real_T pnt[3], int32_T heid, const
  emxArray_real_T *ps, const emxArray_real_T *nrms, const emxArray_int32_T *tris,
  const emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T *fid,
  real_T nc[2], int8_T *loc, real_T *dist, int32_T *proj);
static boolean_T g_eml_strcmp(const char_T a[3]);
static void gen_vander_bivar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static void gen_vander_univar(const emxArray_real_T *us, int32_T degree,
  emxArray_real_T *V);
static boolean_T h_eml_strcmp(const char_T a[3]);
static void ismooth_trimesh_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_boolean_T *isridge, const
  emxArray_int32_T *flabels, boolean_T check_trank, emxArray_real_T *us_smooth);
static void linfit_lhf_grad_surf_point(const int32_T ngbvs[128], const
  emxArray_real_T *us, const real_T t1[3], const real_T t2[3], const
  emxArray_real_T *nrms, const emxArray_real_T *ws, real_T hess[3]);
static real_T norm2_vec(const emxArray_real_T *v);
static void obtain_all_nring_surf(real_T ring, int32_T nv, const
  emxArray_int32_T *elems, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, emxArray_int32_T *allv_ngbvs);
static int32_T obtain_nring_quad(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *elems, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags);

static int32_T c_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags, const int32_T ngbfs[256]);

static void polyfit3d_cmf_edge(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, real_T xi, int32_T deg, real_T pnt[3]);
static void polyfit3d_cmf_tri(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, const emxArray_real_T *ngbpnts3, const emxArray_real_T *nrms3, real_T
  xi, real_T eta, int32_T deg, real_T pnt[3]);
static void polyfit3d_walf_vertex(const emxArray_real_T *pnts, const
  emxArray_real_T *nrms, const real_T pos[3], int32_T deg, real_T pnt[3]);
static void polyfit_lhf_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, const
  emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T
  ring, emxArray_real_T *nrms, emxArray_real_T *curs, emxArray_real_T *prdirs);
static void polyfit_lhf_surf_point(int32_T v, const int32_T ngbvs[128], int32_T
  nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms_coor, int32_T
  degree, real_T nrm[3], int32_T *deg);
static void polyfit_lhfgrad_surf_point(int32_T v, const int32_T ngbvs[128],
  int32_T nverts, const emxArray_real_T *xs, const emxArray_real_T *nrms,
  int32_T degree, int32_T *deg, real_T prcurvs[2]);
static void project_onto_one_ring(const real_T pnt[3], int32_T *fid, int32_T lid,
  const emxArray_real_T *ps, const emxArray_real_T *nrms, const emxArray_int32_T
  *tris, const emxArray_int32_T *opphes, real_T nc[2], int8_T *loc, real_T *dist);
static int32_T qr_safeguarded(emxArray_real_T *A, int32_T ncols, emxArray_real_T
  *D);
static void rdivide(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *z);
static void repmat(real_T m, emxArray_real_T *b);
static void rescale_displacements(const emxArray_real_T *xs, const
  emxArray_real_T *us, const emxArray_int32_T *tris, emxArray_real_T *alpha_vs);
static void rescale_matrix(emxArray_real_T *V, int32_T ncols, emxArray_real_T
  *ts);
static void smoothing_single_iteration(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms, const
  emxArray_int32_T *opphes, const emxArray_boolean_T *isridge, const
  emxArray_boolean_T *ridgeedge, const emxArray_int32_T *flabel, boolean_T
  *scaled, real_T min_angle, boolean_T check_trank, const char_T
  hisurf_args_method[3], int32_T hisurf_args_degree, int32_T verbose,
  emxArray_real_T *us_smooth);
static real_T sum(const emxArray_real_T *x);

/* Function Definitions */

/*
 * function [elem_energies, grads_smooth, Hs_smooth] = accumulate_isometry_energy_tri...
 *     (xs, tris, refareas, mu)
 */
static void accumulate_isometry_energy_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, emxArray_real_T *elem_energies, emxArray_real_T
  *grads_smooth, emxArray_real_T *Hs_smooth)
{
  int32_T iy;
  int32_T ix;
  int32_T jj;
  int32_T b_ix;
  real_T b_xs[9];
  real_T c_xs[9];
  real_T e12[3];
  real_T sql12;
  int32_T b_iy;
  int32_T k;
  real_T e23[3];
  real_T sql23;
  real_T e31[3];
  real_T sql31;
  real_T nrm[3];
  real_T energy_a;
  real_T area2;
  real_T energy;
  real_T e12_orth[3];
  real_T e23_orth[3];
  real_T e31_orth[3];
  real_T c;
  real_T b_e12;
  real_T grads1;
  real_T grads2;
  real_T grads3;
  real_T b_grads3[3];
  real_T b_grads2[3];
  real_T b_grads1[3];
  real_T Hess[27];
  real_T N[9];

  /*  ACCUMULATE_ISOMETRY_ENERGY_TRI   Accumulate energy for isometric mapping */
  /*       for tetrahedral mesh. */
  /* 'accumulate_isometry_energy_tri:6' if nargin<3 */
  /* 'accumulate_isometry_energy_tri:8' withrefarea = mu>0 && any(refareas~=0); */
  /* 'accumulate_isometry_energy_tri:10' grads_smooth = zeros(3,size(xs,1)); */
  iy = grads_smooth->size[0] * grads_smooth->size[1];
  grads_smooth->size[0] = 3;
  emxEnsureCapacity((emxArray__common *)grads_smooth, iy, (int32_T)sizeof(real_T));
  ix = xs->size[0];
  iy = grads_smooth->size[0] * grads_smooth->size[1];
  grads_smooth->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)grads_smooth, iy, (int32_T)sizeof(real_T));
  ix = 3 * xs->size[0] - 1;
  for (iy = 0; iy <= ix; iy++) {
    grads_smooth->data[iy] = 0.0;
  }

  /* 'accumulate_isometry_energy_tri:11' Hs_smooth = zeros(3,3,size(xs,1)); */
  iy = Hs_smooth->size[0] * Hs_smooth->size[1] * Hs_smooth->size[2];
  Hs_smooth->size[0] = 3;
  Hs_smooth->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)Hs_smooth, iy, (int32_T)sizeof(real_T));
  ix = xs->size[0];
  iy = Hs_smooth->size[0] * Hs_smooth->size[1] * Hs_smooth->size[2];
  Hs_smooth->size[2] = ix;
  emxEnsureCapacity((emxArray__common *)Hs_smooth, iy, (int32_T)sizeof(real_T));
  ix = 9 * xs->size[0] - 1;
  for (iy = 0; iy <= ix; iy++) {
    Hs_smooth->data[iy] = 0.0;
  }

  /* 'accumulate_isometry_energy_tri:12' if nargout>2 */
  /* 'accumulate_isometry_energy_tri:12' elem_energies = zeros( size(tris,1),1); */
  iy = elem_energies->size[0];
  elem_energies->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)elem_energies, iy, (int32_T)sizeof
                    (real_T));
  ix = tris->size[0] - 1;
  for (iy = 0; iy <= ix; iy++) {
    elem_energies->data[iy] = 0.0;
  }

  /* 'accumulate_isometry_energy_tri:14' for jj=1:int32(size(tris,1)) */
  for (jj = 0; jj + 1 <= tris->size[0]; jj++) {
    /*  Compute elemental isometric energy */
    /* 'accumulate_isometry_energy_tri:16' if withrefarea */
    /* 'accumulate_isometry_energy_tri:18' else */
    /* 'accumulate_isometry_energy_tri:19' [energy,grads,Hess] = isometry_energy_eqtri( xs(tris(jj,1:3),1:3)); */
    /* ISOMETRY_ENERTY_EQTRI   Compute isometry-based energy against equlateral triangle. */
    /*  */
    /*    [ENERGY, GRAD, HESS, AREA] = ISOMETRY_ENERGY_EQTRI(XS) */
    /*    [ENERGY, GRAD, HESS, AREA] = ISOMETRY_ENERGY_EQTRI(XS, REFAREA, MU) */
    /*    computes the value, gradient, and Hessian of the energy. */
    /*  */
    /*    MU controls the relative importance of angle-preservation versus */
    /*    area preservation. If it is missing, then it is set to zero. */
    /*  */
    /*    See also ISOMETRY_ENERGY_TRI */
    /* 'isometry_energy_eqtri:13' if nargin<3 */
    /* 'isometry_energy_eqtri:13' mu=0; */
    /* 'isometry_energy_eqtri:13' refarea2=0; */
    /*  compute edge lengths */
    /* 'isometry_energy_eqtri:16' e12 = (xs(2,1:3)-xs(1,1:3))'; */
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        b_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        c_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      e12[iy] = b_xs[3 + iy] - c_xs[iy];
    }

    /* 'isometry_energy_eqtri:16' sql12 = e12'*e12; */
    sql12 = 0.0;
    ix = 0;
    b_iy = 0;
    for (k = 0; k < 3; k++) {
      sql12 += e12[ix] * e12[b_iy];
      ix++;
      b_iy++;
    }

    /* 'isometry_energy_eqtri:17' e23 = (xs(3,1:3)-xs(2,1:3))'; */
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        b_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        c_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      e23[iy] = b_xs[6 + iy] - c_xs[3 + iy];
    }

    /* 'isometry_energy_eqtri:17' sql23 = e23'*e23; */
    sql23 = 0.0;
    ix = 0;
    b_iy = 0;
    for (k = 0; k < 3; k++) {
      sql23 += e23[ix] * e23[b_iy];
      ix++;
      b_iy++;
    }

    /* 'isometry_energy_eqtri:18' e31 = (xs(1,1:3)-xs(3,1:3))'; */
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        b_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        c_xs[b_ix + 3 * iy] = xs->data[(tris->data[jj + tris->size[0] * iy] +
          xs->size[0] * b_ix) - 1];
      }
    }

    for (iy = 0; iy < 3; iy++) {
      e31[iy] = b_xs[iy] - c_xs[6 + iy];
    }

    /* 'isometry_energy_eqtri:18' sql31 = e31'*e31; */
    sql31 = 0.0;
    ix = 0;
    b_iy = 0;

    /* 'isometry_energy_eqtri:20' nrm = cross_col(e12, e23); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    nrm[0] = e12[1] * e23[2] - e12[2] * e23[1];
    nrm[1] = e12[2] * e23[0] - e12[0] * e23[2];
    nrm[2] = e12[0] * e23[1] - e12[1] * e23[0];

    /* 'isometry_energy_eqtri:20' area2 = sqrt(nrm'*nrm); */
    energy_a = 0.0;
    b_ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      sql31 += e31[ix] * e31[b_iy];
      ix++;
      b_iy++;
      energy_a += nrm[b_ix] * nrm[iy];
      b_ix++;
      iy++;
    }

    area2 = sqrt(energy_a);

    /* 'isometry_energy_eqtri:21' energy = (sql12 + sql23 + sql31) / area2; */
    energy = ((sql12 + sql23) + sql31) / area2;

    /* 'isometry_energy_eqtri:23' if nargout>1 */
    /* 'isometry_energy_eqtri:24' nrm = nrm/(area2+1.e-100); */
    energy_a = area2 + 1.0E-100;
    for (iy = 0; iy < 3; iy++) {
      nrm[iy] /= energy_a;
    }

    /* 'isometry_energy_eqtri:25' e12_orth = cross_col(nrm, e12); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    e12_orth[0] = nrm[1] * e12[2] - nrm[2] * e12[1];
    e12_orth[1] = nrm[2] * e12[0] - nrm[0] * e12[2];
    e12_orth[2] = nrm[0] * e12[1] - nrm[1] * e12[0];

    /* 'isometry_energy_eqtri:26' e23_orth = cross_col(nrm, e23); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    e23_orth[0] = nrm[1] * e23[2] - nrm[2] * e23[1];
    e23_orth[1] = nrm[2] * e23[0] - nrm[0] * e23[2];
    e23_orth[2] = nrm[0] * e23[1] - nrm[1] * e23[0];

    /* 'isometry_energy_eqtri:27' e31_orth = cross_col(nrm, e31); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    e31_orth[0] = nrm[1] * e31[2] - nrm[2] * e31[1];
    e31_orth[1] = nrm[2] * e31[0] - nrm[0] * e31[2];
    e31_orth[2] = nrm[0] * e31[1] - nrm[1] * e31[0];

    /*  Loop through edges to compute attractive forces */
    /* 'isometry_energy_eqtri:30' c = -2 / area2; */
    c = -2.0 / area2;

    /* 'isometry_energy_eqtri:31' t = c*e12; */
    /*  Loop through edges to compute repulsive forces */
    /* 'isometry_energy_eqtri:36' energy_a = energy / area2; */
    energy_a = energy / area2;
    for (iy = 0; iy < 3; iy++) {
      b_e12 = c * e12[iy];

      /* 'isometry_energy_eqtri:31' grads1 = t; */
      grads1 = b_e12;

      /* 'isometry_energy_eqtri:31' grads2 = -t; */
      grads2 = -b_e12;

      /* 'isometry_energy_eqtri:32' t = c*e23; */
      b_e12 = c * e23[iy];

      /* 'isometry_energy_eqtri:32' grads2 = grads2 + t; */
      grads2 += b_e12;

      /* 'isometry_energy_eqtri:32' grads3 = -t; */
      grads3 = -b_e12;

      /* 'isometry_energy_eqtri:33' t = c*e31; */
      b_e12 = c * e31[iy];

      /* 'isometry_energy_eqtri:33' grads3 = grads3 + t; */
      grads3 += b_e12;

      /* 'isometry_energy_eqtri:33' grads1 = grads1 - t; */
      grads1 -= b_e12;

      /* 'isometry_energy_eqtri:37' grads1 = grads1 - energy_a*e23_orth; */
      grads1 -= energy_a * e23_orth[iy];

      /* 'isometry_energy_eqtri:38' grads2 = grads2 - energy_a*e31_orth; */
      grads2 -= energy_a * e31_orth[iy];

      /* 'isometry_energy_eqtri:39' grads3 = grads3 - energy_a*e12_orth; */
      grads3 -= energy_a * e12_orth[iy];
      b_grads3[iy] = grads3;
      b_grads2[iy] = grads2;
      b_grads1[iy] = grads1;
    }

    /* 'isometry_energy_eqtri:41' if nargout>2 */
    /*         %% Compute Hessian */
    /* 'isometry_energy_eqtri:43' hess = zeros(3,3,3); */
    for (iy = 0; iy < 27; iy++) {
      Hess[iy] = 0.0;
    }

    /* 'isometry_energy_eqtri:45' hess(:,:,1) = neg_outer_pt(grads1/area2,e23_orth); */
    for (ix = 0; ix < 3; ix++) {
      e23[ix] = b_grads1[ix] / area2;
    }

    /*  Compute T = -a*b'-b*a'; */
    /* 'isometry_energy_eqtri:99' coder.inline('always'); */
    /* 'isometry_energy_eqtri:100' T = [-2*a(1)*b(1), -a(1)*b(2)-a(2)*b(1), -a(1)*b(3)-a(3)*b(1); */
    /* 'isometry_energy_eqtri:101'     0, -2*a(2)*b(2), -a(2)*b(3)-a(3)*b(2); */
    /* 'isometry_energy_eqtri:102'     0, 0, -2*a(3)*b(3)]; */
    Hess[0] = -2.0 * e23[0] * e23_orth[0];
    Hess[3] = -e23[0] * e23_orth[1] - e23[1] * e23_orth[0];
    Hess[6] = -e23[0] * e23_orth[2] - e23[2] * e23_orth[0];
    Hess[1] = 0.0;
    Hess[4] = -2.0 * e23[1] * e23_orth[1];
    Hess[7] = -e23[1] * e23_orth[2] - e23[2] * e23_orth[1];
    Hess[2] = 0.0;
    Hess[5] = 0.0;
    Hess[8] = -2.0 * e23[2] * e23_orth[2];

    /* 'isometry_energy_eqtri:46' hess(:,:,2) = neg_outer_pt(grads2/area2,e31_orth); */
    for (ix = 0; ix < 3; ix++) {
      e23[ix] = b_grads2[ix] / area2;
    }

    /*  Compute T = -a*b'-b*a'; */
    /* 'isometry_energy_eqtri:99' coder.inline('always'); */
    /* 'isometry_energy_eqtri:100' T = [-2*a(1)*b(1), -a(1)*b(2)-a(2)*b(1), -a(1)*b(3)-a(3)*b(1); */
    /* 'isometry_energy_eqtri:101'     0, -2*a(2)*b(2), -a(2)*b(3)-a(3)*b(2); */
    /* 'isometry_energy_eqtri:102'     0, 0, -2*a(3)*b(3)]; */
    Hess[9] = -2.0 * e23[0] * e31_orth[0];
    Hess[12] = -e23[0] * e31_orth[1] - e23[1] * e31_orth[0];
    Hess[15] = -e23[0] * e31_orth[2] - e23[2] * e31_orth[0];
    Hess[10] = 0.0;
    Hess[13] = -2.0 * e23[1] * e31_orth[1];
    Hess[16] = -e23[1] * e31_orth[2] - e23[2] * e31_orth[1];
    Hess[11] = 0.0;
    Hess[14] = 0.0;
    Hess[17] = -2.0 * e23[2] * e31_orth[2];

    /* 'isometry_energy_eqtri:47' hess(:,:,3) = neg_outer_pt(grads3/area2,e12_orth); */
    for (ix = 0; ix < 3; ix++) {
      e23[ix] = b_grads3[ix] / area2;
    }

    /*  Compute T = -a*b'-b*a'; */
    /* 'isometry_energy_eqtri:99' coder.inline('always'); */
    /* 'isometry_energy_eqtri:100' T = [-2*a(1)*b(1), -a(1)*b(2)-a(2)*b(1), -a(1)*b(3)-a(3)*b(1); */
    /* 'isometry_energy_eqtri:101'     0, -2*a(2)*b(2), -a(2)*b(3)-a(3)*b(2); */
    /* 'isometry_energy_eqtri:102'     0, 0, -2*a(3)*b(3)]; */
    Hess[18] = -2.0 * e23[0] * e12_orth[0];
    Hess[21] = -e23[0] * e12_orth[1] - e23[1] * e12_orth[0];
    Hess[24] = -e23[0] * e12_orth[2] - e23[2] * e12_orth[0];
    Hess[19] = 0.0;
    Hess[22] = -2.0 * e23[1] * e12_orth[1];
    Hess[25] = -e23[1] * e12_orth[2] - e23[2] * e12_orth[1];
    Hess[20] = 0.0;
    Hess[23] = 0.0;
    Hess[26] = -2.0 * e23[2] * e12_orth[2];

    /* 'isometry_energy_eqtri:49' c = 4/area2; */
    c = 4.0 / area2;

    /* 'isometry_energy_eqtri:50' hess(1,1,:) = hess(1,1,:) + c; */
    /* 'isometry_energy_eqtri:54' N = avvt(-energy_a/area2,nrm); */
    energy_a = -energy_a / area2;
    for (iy = 0; iy < 3; iy++) {
      Hess[9 * iy] += c;

      /* 'isometry_energy_eqtri:51' hess(2,2,:) = hess(2,2,:) + c; */
      Hess[4 + 9 * iy] += c;

      /* 'isometry_energy_eqtri:52' hess(3,3,:) = hess(3,3,:) + c; */
      Hess[8 + 9 * iy] += c;

      /*  Compute N = a*n*n'; */
      /* 'isometry_energy_eqtri:106' coder.inline('always'); */
      /* 'isometry_energy_eqtri:107' n1 = a*n; */
      e23[iy] = energy_a * nrm[iy];
    }

    /* 'isometry_energy_eqtri:108' N = [n1(1)*n(1), n1(1)*n(2), n1(1)*n(3); */
    /* 'isometry_energy_eqtri:109'     0, n1(2)*n(2), n1(2)*n(3); */
    /* 'isometry_energy_eqtri:110'     0, 0, n1(3)*n(3)]; */
    N[0] = e23[0] * nrm[0];
    N[3] = e23[0] * nrm[1];
    N[6] = e23[0] * nrm[2];
    N[1] = 0.0;
    N[4] = e23[1] * nrm[1];
    N[7] = e23[1] * nrm[2];
    N[2] = 0.0;
    N[5] = 0.0;
    N[8] = e23[2] * nrm[2];

    /* 'isometry_energy_eqtri:55' hess(:,:,1) = apby_tensor(hess(:,:,1), sql23, N); */
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        b_xs[b_ix + 3 * iy] = Hess[b_ix + 3 * iy];
      }

      /*  Compute T = T + a * B; */
      /* 'isometry_energy_eqtri:128' coder.inline('always'); */
      /* 'isometry_energy_eqtri:129' T(1,:) = T(1,:)+a*B(1,:); */
      b_xs[3 * iy] = Hess[3 * iy] + sql23 * N[3 * iy];
    }

    /* 'isometry_energy_eqtri:130' T(2,2:3) = T(2,2:3)+a*B(2,2:3); */
    for (iy = 0; iy < 2; iy++) {
      b_xs[1 + 3 * (1 + iy)] += sql23 * N[1 + 3 * (1 + iy)];
    }

    /* 'isometry_energy_eqtri:131' T(3,3) = T(3,3)+a*B(3,3); */
    b_xs[8] += sql23 * N[8];
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        Hess[b_ix + 3 * iy] = b_xs[b_ix + 3 * iy];

        /* 'isometry_energy_eqtri:56' hess(:,:,2) = apby_tensor(hess(:,:,2), sql31, N); */
        b_xs[b_ix + 3 * iy] = Hess[9 + (b_ix + 3 * iy)];
      }

      /*  Compute T = T + a * B; */
      /* 'isometry_energy_eqtri:128' coder.inline('always'); */
      /* 'isometry_energy_eqtri:129' T(1,:) = T(1,:)+a*B(1,:); */
      b_xs[3 * iy] = Hess[9 + 3 * iy] + sql31 * N[3 * iy];
    }

    /* 'isometry_energy_eqtri:130' T(2,2:3) = T(2,2:3)+a*B(2,2:3); */
    for (iy = 0; iy < 2; iy++) {
      b_xs[1 + 3 * (1 + iy)] += sql31 * N[1 + 3 * (1 + iy)];
    }

    /* 'isometry_energy_eqtri:131' T(3,3) = T(3,3)+a*B(3,3); */
    b_xs[8] += sql31 * N[8];
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        Hess[9 + (b_ix + 3 * iy)] = b_xs[b_ix + 3 * iy];

        /* 'isometry_energy_eqtri:57' hess(:,:,3) = apby_tensor(hess(:,:,3), sql12, N); */
        b_xs[b_ix + 3 * iy] = Hess[18 + (b_ix + 3 * iy)];
      }

      /*  Compute T = T + a * B; */
      /* 'isometry_energy_eqtri:128' coder.inline('always'); */
      /* 'isometry_energy_eqtri:129' T(1,:) = T(1,:)+a*B(1,:); */
      b_xs[3 * iy] = Hess[18 + 3 * iy] + sql12 * N[3 * iy];
    }

    /* 'isometry_energy_eqtri:130' T(2,2:3) = T(2,2:3)+a*B(2,2:3); */
    for (iy = 0; iy < 2; iy++) {
      b_xs[1 + 3 * (1 + iy)] += sql12 * N[1 + 3 * (1 + iy)];
    }

    /* 'isometry_energy_eqtri:131' T(3,3) = T(3,3)+a*B(3,3); */
    b_xs[8] += sql12 * N[8];
    for (iy = 0; iy < 3; iy++) {
      for (b_ix = 0; b_ix < 3; b_ix++) {
        Hess[18 + (b_ix + 3 * iy)] = b_xs[b_ix + 3 * iy];
      }

      /* % Compute energy for area-preservation */
      /* 'isometry_energy_eqtri:62' if mu>0 */
      /* 'isometry_energy_eqtri:86' if nargout>1 */
      /* 'isometry_energy_eqtri:87' grads = [grads1, grads2, grads3]; */
      N[iy] = b_grads1[iy];
      N[3 + iy] = b_grads2[iy];
      N[6 + iy] = b_grads3[iy];
    }

    /* 'isometry_energy_eqtri:88' if nargout>2 */
    /* 'isometry_energy_eqtri:89' hess(2,1,:) = hess(1,2,:); */
    /* 'isometry_energy_eqtri:90' hess(3,1,:) = hess(1,3,:); */
    /* 'isometry_energy_eqtri:91' hess(3,2,:) = hess(2,3,:); */
    for (iy = 0; iy < 3; iy++) {
      Hess[1 + 9 * iy] = Hess[3 + 9 * iy];
      Hess[2 + 9 * iy] = Hess[6 + 9 * iy];
      Hess[5 + 9 * iy] = Hess[7 + 9 * iy];
    }

    /* 'isometry_energy_eqtri:95' area = 0.5*area2; */
    /*  Accumulate energy to vertices */
    /* 'accumulate_isometry_energy_tri:23' for kk=1:3 */
    for (ix = 0; ix < 3; ix++) {
      /* 'accumulate_isometry_energy_tri:24' v = tris(jj,kk); */
      /* 'accumulate_isometry_energy_tri:26' grads_smooth(:,v) = grads_smooth(:,v) + grads(:,kk); */
      b_iy = tris->data[jj + tris->size[0] * ix];
      k = tris->data[jj + tris->size[0] * ix];
      for (iy = 0; iy < 3; iy++) {
        e12[iy] = grads_smooth->data[iy + grads_smooth->size[0] * (k - 1)] +
          N[iy + 3 * ix];
      }

      for (iy = 0; iy < 3; iy++) {
        grads_smooth->data[iy + grads_smooth->size[0] * (b_iy - 1)] = e12[iy];
      }

      /* 'accumulate_isometry_energy_tri:27' Hs_smooth(:,:,v)  = Hs_smooth(:,:,v)  + Hess(:,:,kk); */
      b_iy = tris->data[jj + tris->size[0] * ix];
      k = tris->data[jj + tris->size[0] * ix];
      for (iy = 0; iy < 3; iy++) {
        for (b_ix = 0; b_ix < 3; b_ix++) {
          b_xs[b_ix + 3 * iy] = Hs_smooth->data[(b_ix + Hs_smooth->size[0] * iy)
            + Hs_smooth->size[0] * Hs_smooth->size[1] * (k - 1)] + Hess[(b_ix +
            3 * iy) + 9 * ix];
        }
      }

      for (iy = 0; iy < 3; iy++) {
        for (b_ix = 0; b_ix < 3; b_ix++) {
          Hs_smooth->data[(b_ix + Hs_smooth->size[0] * iy) + Hs_smooth->size[0] *
            Hs_smooth->size[1] * (b_iy - 1)] = b_xs[b_ix + 3 * iy];
        }
      }
    }

    /*  Save energy */
    /* 'accumulate_isometry_energy_tri:31' elem_energies(jj) = energy; */
    elem_energies->data[jj] = energy;
  }
}

/*
 * function [pnt_added,xs] = add_disps_to_pntpositions(nv_clean, nt_clean, xs, tris,...
 * us_smooth, min_angle_pre, angletol_max, scaled)
 */
static boolean_T add_disps_to_pntpositions(int32_T nv_clean, int32_T nt_clean,
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *us_smooth, real_T min_angle_pre, boolean_T scaled)
{
  boolean_T pnt_added;
  emxArray_real_T *b_xs;
  emxArray_real_T *c_xs;
  int32_T i19;
  real_T max_area;
  real_T min_area;
  real_T max_angle;
  real_T min_angle;
  int32_T i20;
  int32_T loop_ub;
  int32_T i21;

  /* 'add_disps_to_pntpositions:3' coder.inline('never') */
  /* 'add_disps_to_pntpositions:4' pnt_added = false; */
  pnt_added = FALSE;

  /* 'add_disps_to_pntpositions:5' if min_angle_pre>=angletol_max && ~scaled */
  emxInit_real_T(&b_xs, 2);
  emxInit_real_T(&c_xs, 2);
  if ((min_angle_pre >= 27.0) && (!scaled)) {
    /*  Step 1: Compute the new positions by adding the displacements */
    /* 'add_disps_to_pntpositions:7' xs_new = xs(1:nv_clean,:) + us_smooth(1:nv_clean,:); */
    if (1 > nv_clean) {
      i19 = 0;
    } else {
      i19 = nv_clean;
    }

    /*  Step 2: Compute the minimum angle of the new mesh     */
    /* 'add_disps_to_pntpositions:10' [min_angle, max_angle, min_area, max_area] = compute_statistics_tris_global(nt_clean, xs, tris); */
    compute_statistics_tris_global(nt_clean, xs, tris, &min_angle, &max_angle,
      &min_area, &max_area);

    /* 'add_disps_to_pntpositions:12' if min_angle>min_angle_pre */
    if (min_angle > min_angle_pre) {
      /* 'add_disps_to_pntpositions:13' xs(1:nv_clean,:) = xs_new; */
      i20 = c_xs->size[0] * c_xs->size[1];
      c_xs->size[0] = i19;
      c_xs->size[1] = 3;
      emxEnsureCapacity((emxArray__common *)c_xs, i20, (int32_T)sizeof(real_T));
      for (i20 = 0; i20 < 3; i20++) {
        loop_ub = i19 - 1;
        for (i21 = 0; i21 <= loop_ub; i21++) {
          c_xs->data[i21 + c_xs->size[0] * i20] = xs->data[i21 + xs->size[0] *
            i20] + us_smooth->data[i21 + us_smooth->size[0] * i20];
        }
      }

      for (i19 = 0; i19 < 3; i19++) {
        loop_ub = c_xs->size[0] - 1;
        for (i20 = 0; i20 <= loop_ub; i20++) {
          xs->data[i20 + xs->size[0] * i19] = c_xs->data[i20 + c_xs->size[0] *
            i19];
        }
      }

      /* 'add_disps_to_pntpositions:14' pnt_added = true; */
      pnt_added = TRUE;
    }
  } else {
    if (!scaled) {
      /* 'add_disps_to_pntpositions:16' elseif ~scaled */
      /* 'add_disps_to_pntpositions:17' xs(1:nv_clean,:)= xs(1:nv_clean,:) + us_smooth(1:nv_clean,:); */
      if (1 > nv_clean) {
        i19 = 0;
      } else {
        i19 = nv_clean;
      }

      i20 = b_xs->size[0] * b_xs->size[1];
      b_xs->size[0] = i19;
      b_xs->size[1] = 3;
      emxEnsureCapacity((emxArray__common *)b_xs, i20, (int32_T)sizeof(real_T));
      for (i20 = 0; i20 < 3; i20++) {
        loop_ub = i19 - 1;
        for (i21 = 0; i21 <= loop_ub; i21++) {
          b_xs->data[i21 + b_xs->size[0] * i20] = xs->data[i21 + xs->size[0] *
            i20] + us_smooth->data[i21 + us_smooth->size[0] * i20];
        }
      }

      for (i19 = 0; i19 < 3; i19++) {
        loop_ub = b_xs->size[0] - 1;
        for (i20 = 0; i20 <= loop_ub; i20++) {
          xs->data[i20 + xs->size[0] * i19] = b_xs->data[i20 + b_xs->size[0] *
            i19];
        }
      }

      /* 'add_disps_to_pntpositions:18' pnt_added = true; */
      pnt_added = TRUE;
    }
  }

  emxFree_real_T(&c_xs);
  emxFree_real_T(&b_xs);
  return pnt_added;
}

/*
 *
 */
static boolean_T any(const boolean_T x[3])
{
  boolean_T y;
  int32_T k;
  boolean_T exitg1;
  boolean_T b0;
  y = FALSE;
  k = 1;
  exitg1 = 0U;
  while ((exitg1 == 0U) && (k <= 3)) {
    if (x[k - 1] == 0) {
      b0 = TRUE;
    } else {
      b0 = FALSE;
    }

    if (!b0) {
      y = TRUE;
      exitg1 = 1U;
    } else {
      k++;
    }
  }

  return y;
}

/*
 * function [us_smooth, scaled] = async_scale_disps_tri_cleanmesh(nv_clean,xs, us_smooth, tris, tol)
 */
static boolean_T async_scale_disps_tri_cleanmesh(int32_T nv_clean, const
  emxArray_real_T *xs, emxArray_real_T *us_smooth, const emxArray_int32_T *tris)
{
  boolean_T scaled;
  emxArray_real_T *alpha_tmp;
  int32_T niter;
  emxArray_int32_T *r18;
  emxArray_int32_T *r19;
  emxArray_real_T *b_alpha_tmp;
  emxArray_real_T *c_alpha_tmp;
  emxArray_real_T *d_alpha_tmp;
  int32_T exitg1;
  int32_T i16;
  int32_T unnamed_idx_1;
  int32_T loop_ub;
  int32_T exitg2;
  boolean_T b;
  b_emxInit_real_T(&alpha_tmp, 1);

  /*  ASYNC_RESCALE_DISP_TRI    Asynchronously rescale tangential displacements. */
  /* 'async_scale_disps_tri_cleanmesh:6' coder.inline('never') */
  /* 'async_scale_disps_tri_cleanmesh:8' if nargin<5 */
  /* 'async_scale_disps_tri_cleanmesh:10' [alpha_tmp,us_smooth] = rescale_displacements(xs, us_smooth, tris, tol); */
  rescale_displacements(xs, us_smooth, tris, alpha_tmp);

  /* 'async_scale_disps_tri_cleanmesh:12' niter = 0; */
  niter = 0;

  /* 'async_scale_disps_tri_cleanmesh:12' assert(~isempty(alpha_tmp)); */
  /* 'async_scale_disps_tri_cleanmesh:14' scaled = false; */
  scaled = FALSE;

  /* 'async_scale_disps_tri_cleanmesh:15' while anylessthan(alpha_tmp(1:nv_clean), 1) */
  b_emxInit_int32_T(&r18, 2);
  emxInit_int32_T(&r19, 1);
  b_emxInit_real_T(&b_alpha_tmp, 1);
  b_emxInit_real_T(&c_alpha_tmp, 1);
  b_emxInit_real_T(&d_alpha_tmp, 1);
  do {
    exitg1 = 0U;
    if (1 > nv_clean) {
      i16 = 0;
    } else {
      i16 = nv_clean;
    }

    unnamed_idx_1 = r19->size[0];
    r19->size[0] = i16;
    emxEnsureCapacity((emxArray__common *)r19, unnamed_idx_1, (int32_T)sizeof
                      (int32_T));
    loop_ub = i16 - 1;
    for (i16 = 0; i16 <= loop_ub; i16++) {
      r19->data[i16] = 1 + i16;
    }

    i16 = r18->size[0] * r18->size[1];
    r18->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)r18, i16, (int32_T)sizeof(int32_T));
    unnamed_idx_1 = r19->size[0];
    i16 = r18->size[0] * r18->size[1];
    r18->size[1] = unnamed_idx_1;
    emxEnsureCapacity((emxArray__common *)r18, i16, (int32_T)sizeof(int32_T));
    loop_ub = r19->size[0] - 1;
    for (i16 = 0; i16 <= loop_ub; i16++) {
      r18->data[i16] = r19->data[i16];
    }

    /*  Check whether any value within vector v is less than alpha. */
    /* 'anylessthan:4' for i=1:int32(length(v)) */
    unnamed_idx_1 = r18->size[1];
    i16 = b_alpha_tmp->size[0];
    b_alpha_tmp->size[0] = unnamed_idx_1;
    emxEnsureCapacity((emxArray__common *)b_alpha_tmp, i16, (int32_T)sizeof
                      (real_T));
    loop_ub = unnamed_idx_1 - 1;
    for (i16 = 0; i16 <= loop_ub; i16++) {
      b_alpha_tmp->data[i16] = alpha_tmp->data[r18->data[i16] - 1];
    }

    if (b_alpha_tmp->size[0] == 0) {
      loop_ub = 0;
    } else {
      unnamed_idx_1 = r18->size[1];
      i16 = c_alpha_tmp->size[0];
      c_alpha_tmp->size[0] = unnamed_idx_1;
      emxEnsureCapacity((emxArray__common *)c_alpha_tmp, i16, (int32_T)sizeof
                        (real_T));
      loop_ub = unnamed_idx_1 - 1;
      for (i16 = 0; i16 <= loop_ub; i16++) {
        c_alpha_tmp->data[i16] = alpha_tmp->data[r18->data[i16] - 1];
      }

      if (c_alpha_tmp->size[0] > 1) {
        unnamed_idx_1 = r18->size[1];
        i16 = d_alpha_tmp->size[0];
        d_alpha_tmp->size[0] = unnamed_idx_1;
        emxEnsureCapacity((emxArray__common *)d_alpha_tmp, i16, (int32_T)sizeof
                          (real_T));
        loop_ub = unnamed_idx_1 - 1;
        for (i16 = 0; i16 <= loop_ub; i16++) {
          d_alpha_tmp->data[i16] = alpha_tmp->data[r18->data[i16] - 1];
        }

        loop_ub = d_alpha_tmp->size[0];
      } else {
        loop_ub = 1;
      }
    }

    unnamed_idx_1 = 1;
    do {
      exitg2 = 0U;
      if (unnamed_idx_1 <= loop_ub) {
        /* 'anylessthan:5' if (v(i)<alpha) */
        if (alpha_tmp->data[r18->data[unnamed_idx_1 - 1] - 1] < 1.0) {
          /* 'anylessthan:5' b = true; */
          b = TRUE;
          exitg2 = 1U;
        } else {
          unnamed_idx_1++;
        }
      } else {
        /* 'anylessthan:7' b = false; */
        b = FALSE;
        exitg2 = 1U;
      }
    } while (exitg2 == 0U);

    if (b) {
      /* 'async_scale_disps_tri_cleanmesh:16' scaled = true; */
      scaled = TRUE;

      /* 'async_scale_disps_tri_cleanmesh:17' niter = niter+1; */
      niter++;

      /* 'async_scale_disps_tri_cleanmesh:18' if (niter>10) */
      if (niter > 10) {
        /* 'async_scale_disps_tri_cleanmesh:19' us_smooth(:) = 0; */
        i16 = us_smooth->size[0] * us_smooth->size[1];
        us_smooth->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)us_smooth, i16, (int32_T)sizeof
                          (real_T));
        for (i16 = 0; i16 < 3; i16++) {
          loop_ub = us_smooth->size[0] - 1;
          for (unnamed_idx_1 = 0; unnamed_idx_1 <= loop_ub; unnamed_idx_1++) {
            us_smooth->data[unnamed_idx_1 + us_smooth->size[0] * i16] = 0.0;
          }
        }

        exitg1 = 1U;
      } else {
        /*  Step 1: Scale the displacements  */
        /* 'async_scale_disps_tri_cleanmesh:23' for ii=1:nv_clean */
        for (unnamed_idx_1 = 0; unnamed_idx_1 + 1 <= nv_clean; unnamed_idx_1++)
        {
          /* 'async_scale_disps_tri_cleanmesh:24' us_smooth(ii,1) = us_smooth(ii,1) * alpha_tmp(ii); */
          us_smooth->data[unnamed_idx_1] *= alpha_tmp->data[unnamed_idx_1];

          /* 'async_scale_disps_tri_cleanmesh:25' us_smooth(ii,2) = us_smooth(ii,2) * alpha_tmp(ii); */
          us_smooth->data[unnamed_idx_1 + us_smooth->size[0]] *= alpha_tmp->
            data[unnamed_idx_1];

          /* 'async_scale_disps_tri_cleanmesh:26' us_smooth(ii,3) = us_smooth(ii,3) * alpha_tmp(ii); */
          us_smooth->data[unnamed_idx_1 + (us_smooth->size[0] << 1)] *=
            alpha_tmp->data[unnamed_idx_1];
        }

        /*  Step 2: Update the variables 1. scaled "us_smooth" , 2. alpha_tmp */
        /*  of "nv_clean" pnts and communicate values of ghost pnts (>nv_clean)  */
        /*  Step 3: Again check if any rescaling has to be performed or not. */
        /* 'async_scale_disps_tri_cleanmesh:33' [alpha_tmp,us_smooth] = rescale_displacements(xs, us_smooth, tris, tol, alpha_tmp); */
        b_rescale_displacements(xs, us_smooth, tris, alpha_tmp);
      }
    } else {
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&d_alpha_tmp);
  emxFree_real_T(&c_alpha_tmp);
  emxFree_real_T(&b_alpha_tmp);
  emxFree_int32_T(&r19);
  emxFree_int32_T(&r18);
  emxFree_real_T(&alpha_tmp);
  return scaled;
}

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

/*
 * function bs = backsolve(R, bs, cend, ws)
 */
static void b_backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
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

/*
 * function bs = compute_qtb( Q, bs, ncols)
 */
static void b_compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
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
 * function v2he = determine_incident_halfedges(elems, opphes, v2he)
 */
static void b_determine_incident_halfedges(const emxArray_int32_T *elems, const
  emxArray_int32_T *opphes, emxArray_int32_T *v2he)
{
  int32_T kk;
  int32_T loop_ub;
  boolean_T guard1 = FALSE;
  uint32_T a;

  /* DETERMINE_INCIDENT_HALFEDGES Determine an incident halfedges. */
  /*  DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) Determines incident */
  /*  halfedges of each vertex for a triangular, quadrilateral, or mixed mesh.  */
  /*  It gives higher priorities to border edges. The following explains inputs */
  /*  and outputs. */
  /*  */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*      ELEMS is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      OPPHES is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      V2HE is an array of size equal to number of vertices. */
  /*           It is passed by reference. */
  /*  */
  /*  See also DETERMINE_INCIDENT_HALFFACES, DETERMINE_INCIDENT_HALFVERTS */
  /* 'determine_incident_halfedges:18' coder.inline('never'); */
  /*  assert((size(elems,2)==3) && (size(elems,1)>=1) && isa(elems,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /*  assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /* 'determine_incident_halfedges:23' if nargin<3 */
  /* 'determine_incident_halfedges:35' else */
  /* 'determine_incident_halfedges:36' v2he(:) = 0; */
  kk = v2he->size[0];
  emxEnsureCapacity((emxArray__common *)v2he, kk, (int32_T)sizeof(int32_T));
  loop_ub = v2he->size[0] - 1;
  for (kk = 0; kk <= loop_ub; kk++) {
    v2he->data[kk] = 0;
  }

  /* 'determine_incident_halfedges:39' for kk=1:int32(size(elems,1)) */
  kk = 0;
  while ((kk + 1 <= elems->size[0]) && (!(elems->data[kk] == 0))) {
    /* 'determine_incident_halfedges:40' if elems(kk,1)==0 */
    /* 'determine_incident_halfedges:42' for lid=1:int32(size(elems,2)) */
    for (loop_ub = 0; loop_ub < 3; loop_ub++) {
      /* 'determine_incident_halfedges:43' v = elems(kk,lid); */
      /* 'determine_incident_halfedges:44' if v>0 && (v2he(v)==0 || opphes( kk,lid) == 0 || ... */
      /* 'determine_incident_halfedges:45' 	     (opphes( int32( bitshift( uint32(v2he(v)),-2)), mod(v2he(v),4)+1) && opphes( kk, lid)<0)) */
      if (elems->data[kk + elems->size[0] * loop_ub] > 0) {
        guard1 = FALSE;
        if ((v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] == 0) ||
            (opphes->data[kk + opphes->size[0] * loop_ub] == 0)) {
          guard1 = TRUE;
        } else {
          a = (uint32_T)v2he->data[elems->data[kk + elems->size[0] * loop_ub] -
            1];
          if ((opphes->data[((int32_T)(a >> 2U) + opphes->size[0] * (v2he->
                 data[elems->data[kk + elems->size[0] * loop_ub] - 1] -
                 ((v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] >>
                   2) << 2))) - 1] != 0) && (opphes->data[kk + opphes->size[0] *
               loop_ub] < 0)) {
            guard1 = TRUE;
          }
        }

        if (guard1 == TRUE) {
          /* 'determine_incident_halfedges:46' v2he(v) = 4*kk + lid - 1; */
          v2he->data[elems->data[kk + elems->size[0] * loop_ub] - 1] = ((kk + 1)
            << 2) + loop_ub;
        }
      }
    }

    kk++;
  }
}

/*
 * function [us, Vs, tranks, lambdas] = eigenanalysis_surf( As, bs, isridge, us, to_update)
 */
static void b_eigenanalysis_surf(const emxArray_real_T *As, const
  emxArray_real_T *bs, const emxArray_boolean_T *isridge, emxArray_real_T *us,
  emxArray_real_T *Vs)
{
  int32_T nv;
  int32_T ix;
  int32_T jj;
  int32_T iy;
  real_T b_As[9];
  real_T ls[3];
  real_T V[9];
  real_T a[3];
  real_T b_sign;
  int32_T k;
  real_T y;
  real_T d[3];
  boolean_T guard1 = FALSE;

  /* EIGENANALYSIS_SURF   Performs eigenvalue decomposition of normal tensor. */
  /*   US = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS] = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS, TRANKS] = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS, TRANKS, LAMBDAS] = EIGENANALYSIS_SURF( AS, BS) */
  /*        solves normal displacement vector and saves into US. */
  /*        Eigenvectors and eignvalues are stored into VS and LAMBDAS. */
  /*        Tangent ranks are stored into TRANKS. */
  /*  */
  /*   EIGENANALYSIS_SURF( AS, BS, ISRIDGE) */
  /*        also considers the input flags for ridge vertices. */
  /*  */
  /*   EIGENANALYSIS_SURF( AS, BS, ISRIDGE, US, TO_UPDATE) */
  /*        updates us only for vertices with to_update set to a nonzero value. */
  /*  */
  /*   See also COMPUTE_MEDIAL_QUADRIC_SURF, COMPUTE_OFFSET_QUADRIC_SURF. */
  /* # coder.typeof(false,[inf,1],[1,0]),coder.typeof(0,[inf,3],[1,0]), */
  /* # coder.typeof(false,[inf,1],[1,0])} */
  /* 'eigenanalysis_surf:23' nv = int32(size(As,3)); */
  nv = As->size[2];

  /* 'eigenanalysis_surf:25' if nargin<4 */
  /* 'eigenanalysis_surf:25' us = coder.nullcopy(zeros(nv,3)); */
  ix = us->size[0] * us->size[1];
  us->size[0] = nv;
  us->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

  /* 'eigenanalysis_surf:27' if nargout>1 */
  /* 'eigenanalysis_surf:27' Vs = coder.nullcopy(zeros(3,3,nv)); */
  ix = Vs->size[0] * Vs->size[1] * Vs->size[2];
  Vs->size[0] = 3;
  Vs->size[1] = 3;
  Vs->size[2] = nv;
  emxEnsureCapacity((emxArray__common *)Vs, ix, (int32_T)sizeof(real_T));

  /* 'eigenanalysis_surf:28' if nargout>2 */
  /*  Tangent ranks */
  /* 'eigenanalysis_surf:29' if nargout>3 */
  /* 'eigenanalysis_surf:31' tol = 0.003; */
  /*  Loop through vertices one by one to compute displacements */
  /* 'eigenanalysis_surf:34' for jj=1:nv */
  for (jj = 0; jj + 1 <= nv; jj++) {
    /* 'eigenanalysis_surf:35' if nargin==5 && ~to_update(jj) */
    /* 'eigenanalysis_surf:37' b = bs(1:3,jj); */
    /* 'eigenanalysis_surf:38' if nargout==1 && all(b==0) */
    /*  Perform eigenvalue decomposition and sort eigenvalues and eigenvectors */
    /* 'eigenanalysis_surf:41' A = As(1:3,1:3,jj); */
    /* 'eigenanalysis_surf:42' [V,ls] = eig3_sorted(A); */
    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        b_As[iy + 3 * ix] = As->data[(iy + As->size[0] * ix) + As->size[0] *
          As->size[1] * jj];
      }
    }

    eig3_sorted(b_As, V, ls);

    /* 'eigenanalysis_surf:43' if ls(1)==0 */
    if (ls[0] == 0.0) {
    } else {
      /* 'eigenanalysis_surf:45' sign = V(1:3,1)'*b; */
      for (ix = 0; ix < 3; ix++) {
        a[ix] = V[ix];
      }

      b_sign = 0.0;
      ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        b_sign += a[ix] * bs->data[iy + bs->size[0] * jj];
        ix++;
        iy++;
      }

      /* 'eigenanalysis_surf:46' d = sign/ls(1)*V(1:3,1); */
      y = b_sign / ls[0];
      for (ix = 0; ix < 3; ix++) {
        d[ix] = y * V[ix];
      }

      /* 'eigenanalysis_surf:48' trank = int8(2); */
      /* 'eigenanalysis_surf:50' if nargin>2 && size(isridge,1)>=nv && isridge(jj) && ls(2)>ls(1)*1.e-8 || ... */
      /* 'eigenanalysis_surf:51'             ls(2)>ls(1)*tol || abs(sign)<0.7*sqrt(b'*b) */
      guard1 = FALSE;
      if (((isridge->size[0] >= nv) && isridge->data[jj] && (ls[1] > ls[0] *
            1.0E-8)) || (ls[1] > ls[0] * 0.003)) {
        guard1 = TRUE;
      } else {
        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += bs->data[ix + bs->size[0] * jj] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        if (fabs(b_sign) < 0.7 * sqrt(y)) {
          guard1 = TRUE;
        }
      }

      if (guard1 == TRUE) {
        /*  Ridge vertex */
        /* 'eigenanalysis_surf:52' d = d + (V(1:3,2)'*b)/ls(2)*V(1:3,2); */
        for (ix = 0; ix < 3; ix++) {
          a[ix] = V[3 + ix];
        }

        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += a[ix] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        y /= ls[1];
        for (ix = 0; ix < 3; ix++) {
          d[ix] += y * V[3 + ix];
        }

        /* 'eigenanalysis_surf:53' trank = int8(1); */
      }

      /* 'eigenanalysis_surf:56' if  ls(3)>ls(1)*tol */
      if (ls[2] > ls[0] * 0.003) {
        /*  Corner */
        /* 'eigenanalysis_surf:57' d = d + (V(1:3,3)'*b)/ls(3)*V(1:3,3); */
        for (ix = 0; ix < 3; ix++) {
          a[ix] = V[6 + ix];
        }

        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += a[ix] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        y /= ls[2];
        for (ix = 0; ix < 3; ix++) {
          d[ix] += y * V[6 + ix];
        }

        /* 'eigenanalysis_surf:58' trank = int8(0); */
      }

      /* 'eigenanalysis_surf:61' us(jj,1:3) = d'; */
      for (ix = 0; ix < 3; ix++) {
        us->data[jj + us->size[0] * ix] = d[ix];
      }

      /* 'eigenanalysis_surf:63' if nargout>1 */
      /* 'eigenanalysis_surf:63' Vs(1:3,1:3,jj)=V; */
      for (ix = 0; ix < 3; ix++) {
        for (iy = 0; iy < 3; iy++) {
          Vs->data[(iy + Vs->size[0] * ix) + Vs->size[0] * Vs->size[1] * jj] =
            V[iy + 3 * ix];
        }
      }

      /* 'eigenanalysis_surf:64' if nargout>2 */
      /* 'eigenanalysis_surf:65' if nargout>3 */
    }
  }
}

/*
 *
 */
static boolean_T b_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv1[4] = { 'W', 'A', 'L', 'F' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(3 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv1[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
}

static void b_emxInit_boolean_T(emxArray_boolean_T **pEmxArray, int32_T
  numDimensions)
{
  emxArray_boolean_T *emxArray;
  int32_T loop_ub;
  int32_T i;
  *pEmxArray = (emxArray_boolean_T *)malloc(sizeof(emxArray_boolean_T));
  emxArray = *pEmxArray;
  emxArray->data = (boolean_T *)NULL;
  emxArray->numDimensions = numDimensions;
  emxArray->size = (int32_T *)malloc((uint32_T)(sizeof(int32_T) * numDimensions));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = TRUE;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    emxArray->size[i] = 0;
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
 * function [bs, degree] = eval_vander_bivar( us, bs, degree, ws, interp0, guardosc)
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
  int32_T i12;
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
  /* 'eval_vander_bivar:10' degree = int32(degree); */
  degree = 1;

  /* 'eval_vander_bivar:11' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:14' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:15' if nargin<5 */
  /* 'eval_vander_bivar:16' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:19' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  /* 'eval_vander_bivar:20' while npnts<ncols && degree>1 */
  /* % Construct matrix */
  /* 'eval_vander_bivar:26' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, 1, V);

  /* 'eval_vander_bivar:27' if interp0 */
  /* 'eval_vander_bivar:27' V=V(:,2:end); */
  ii = V->size[1];
  if (2 > ii) {
    jj = 0;
    ii = 0;
  } else {
    jj = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i12 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = ii - jj;
  emxEnsureCapacity((emxArray__common *)b_V, i12, (int32_T)sizeof(real_T));
  loop_ub = (ii - jj) - 1;
  for (ii = 0; ii <= loop_ub; ii++) {
    b_loop_ub = c_V - 1;
    for (i12 = 0; i12 <= b_loop_ub; i12++) {
      b_V->data[i12 + b_V->size[0] * ii] = V->data[i12 + V->size[0] * (jj + ii)];
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
  /* 'eval_vander_bivar:30' if nargin>3 && ~isempty(ws) */
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:31' if degree>2 */
    /* 'eval_vander_bivar:55' else */
    /* 'eval_vander_bivar:56' for ii=1:npnts */
    for (ii = 0; ii + 1 <= npnts; ii++) {
      /* 'eval_vander_bivar:57' for jj=1:ncols */
      for (jj = 0; jj + 1 < 3; jj++) {
        /* 'eval_vander_bivar:57' V(ii,jj) = V(ii,jj) * ws(ii); */
        V->data[ii + V->size[0] * jj] *= ws->data[ii];
      }

      /* 'eval_vander_bivar:58' for jj=1:int32(size(bs,2)) */
      for (jj = 0; jj < 2; jj++) {
        /* 'eval_vander_bivar:58' bs(ii,jj) = bs(ii,jj) * ws(ii); */
        bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
      }
    }
  }

  b_emxInit_real_T(&ws1, 1);
  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:65' ts = coder.nullcopy(zeros(ncols,1)); */
  ii = ws1->size[0];
  ws1->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:66' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, 2, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:69' D = coder.nullcopy(zeros(ncols,1)); */
  ii = D->size[0];
  D->size[0] = 2;
  emxEnsureCapacity((emxArray__common *)D, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:70' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  ii = qr_safeguarded(V, 2, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:73' ncols_sub = ncols; */
  /* 'eval_vander_bivar:74' while rnk < ncols_sub */
  if (ii < 2) {
    /* 'eval_vander_bivar:75' degree = degree-1; */
    degree = 0;

    /* 'eval_vander_bivar:77' if degree==0 */
    /*  Matrix is singular. Consider surface as flat. */
    /* 'eval_vander_bivar:79' bs(:) = 0; */
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
    /* 'eval_vander_bivar:85' bs = compute_qtb( V, bs, ncols_sub); */
    b_compute_qtb(V, bs, 2);

    /* % Perform backward substitution and scale the solutions. */
    /* 'eval_vander_bivar:88' for i=1:ncols_sub */
    for (ii = 0; ii + 1 < 3; ii++) {
      /* 'eval_vander_bivar:88' V(i,i) = D(i); */
      V->data[ii + V->size[0] * ii] = D->data[ii];
    }

    /* 'eval_vander_bivar:89' if guardosc */
    /* 'eval_vander_bivar:91' else */
    /* 'eval_vander_bivar:92' bs = backsolve(V, bs, ncols_sub, ts); */
    b_backsolve(V, bs, 2, ws1);
  }

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
  return degree;
}

/*
 * function [bs, deg_out,deg_pnt,deg_qr] = eval_vander_bivar_cmf( us, bs, degree, ws, interp0, safeguard)
 */
static int32_T b_eval_vander_bivar_cmf(const emxArray_real_T *us,
  emxArray_real_T *bs, int32_T degree, const emxArray_real_T *ws)
{
  int32_T deg_out;
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T ii;
  int32_T loop_ub;
  uint32_T jj;
  int32_T b_bs[2];
  int32_T c_bs[2];
  emxArray_real_T d_bs;
  emxArray_real_T e_bs;
  emxArray_real_T *ts;
  emxArray_real_T *D;
  int32_T exitg1;

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
  /* 'eval_vander_bivar_cmf:14' coder.extrinsic('fprintf') */
  /* 'eval_vander_bivar_cmf:16' assert(isa(degree,'int32')); */
  /* 'eval_vander_bivar_cmf:17' if nargin>4 */
  /* 'eval_vander_bivar_cmf:17' assert( isa( interp0, 'logical')); */
  /* 'eval_vander_bivar_cmf:18' if nargin>5 */
  /* 'eval_vander_bivar_cmf:18' assert( isa( safeguard, 'logical')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar_cmf:21' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar_cmf:22' interp0 = (nargin>4 && interp0); */
  /* 'eval_vander_bivar_cmf:23' if nargin<6 */
  /*  Declaring the degree of output */
  /* 'eval_vander_bivar_cmf:26' deg_out = coder.nullcopy(zeros(1,size(bs,2),'int32')); */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar_cmf:30' ncols = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
  ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);

  /* 'eval_vander_bivar_cmf:31' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (degree > 1)) {
    /* 'eval_vander_bivar_cmf:32' degree=degree-1; */
    degree--;

    /* 'eval_vander_bivar_cmf:33' ncols = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
    ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
  }

  emxInit_real_T(&V, 2);

  /* 'eval_vander_bivar_cmf:35' deg_pnt= degree; */
  /* % Construct matrix */
  /* 'eval_vander_bivar_cmf:37' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, degree, V);

  /* 'eval_vander_bivar_cmf:38' if interp0 */
  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar_cmf:41' if nargin>3 && ~isempty(ws) */
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar_cmf:42' for ii=1:npnts */
    for (ii = 0; ii + 1 <= npnts; ii++) {
      /* 'eval_vander_bivar_cmf:43' for jj=1:size(V,2) */
      loop_ub = V->size[1];
      for (jj = 1U; (real_T)jj <= (real_T)loop_ub; jj++) {
        /* 'eval_vander_bivar_cmf:44' V(ii,jj) = V(ii,jj) * ws(ii); */
        V->data[ii + V->size[0] * ((int32_T)jj - 1)] *= ws->data[ii];
      }

      /* 'eval_vander_bivar_cmf:46' for jj=1:size(bs,2) */
      /* 'eval_vander_bivar_cmf:47' bs(ii,jj) = bs(ii,jj) .* ws(ii); */
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

  b_emxInit_real_T(&ts, 1);
  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar_cmf:53' ts = coder.nullcopy(zeros(ncols,1)); */
  npnts = ts->size[0];
  ts->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ts, npnts, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar_cmf:54' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ts);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar_cmf:57' D = coder.nullcopy(zeros(ncols,1)); */
  npnts = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, npnts, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar_cmf:58' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  npnts = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar_cmf:61' ncols_sub = ncols; */
  /* 'eval_vander_bivar_cmf:62' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (npnts < ncols) {
      /* 'eval_vander_bivar_cmf:63' degree = degree-1; */
      degree--;

      /* 'eval_vander_bivar_cmf:64' if degree==0 */
      if (degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar_cmf:66' bs(:) = 0; */
        npnts = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, npnts, (int32_T)sizeof(real_T));
        loop_ub = bs->size[0] - 1;
        for (npnts = 0; npnts <= loop_ub; npnts++) {
          bs->data[npnts] = 0.0;
        }

        exitg1 = 1U;
      } else {
        /* 'eval_vander_bivar_cmf:68' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
      }
    } else {
      /* 'eval_vander_bivar_cmf:70' deg_qr = degree; */
      /* % Compute Q'bs */
      /* 'eval_vander_bivar_cmf:72' bs = compute_qtb( V, bs, ncols_sub); */
      compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar_cmf:75' for i=1:ncols_sub */
      for (npnts = 0; npnts + 1 <= ncols; npnts++) {
        /* 'eval_vander_bivar_cmf:75' V(i,i) = D(i); */
        V->data[npnts + V->size[0] * npnts] = D->data[npnts];
      }

      /* 'eval_vander_bivar_cmf:76' if safeguard */
      /* 'eval_vander_bivar_cmf:77' [bs, deg_out] = backsolve_bivar_safeguarded(V, bs, degree, interp0, ts); */
      deg_out = backsolve_bivar_safeguarded(V, bs, degree, ts);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ts);
  emxFree_real_T(&V);
  return deg_out;
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
  boolean_T b5;
  int32_T fid_in;
  static const int8_T iv30[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv31[3] = { 3, 1, 2 };

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
  boolean_T b6;
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
      b5 = TRUE;
    } else {
      b5 = FALSE;
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
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv30[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      if (!b5) {
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
      lid = iv31[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        if (!b5) {
          /*  Save starting position for next vertex */
          /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv31[lid] -
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
                                  (iv31[lid] - 1)] - 1]) && (nverts >= 128)) ||
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
                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv31[lid] -
                      1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv31[lid]
                    - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  vtags->data[tris->data[fid + tris->size[0] * (iv31[lid] - 1)]
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
              b6 = TRUE;
            } else {
              b6 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b6) {
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
                                (iv30[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                nverts++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv30[lid]
                  - 1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                vtags->data[tris->data[fid + tris->size[0] * (iv30[lid] - 1)] -
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
              lid = iv31[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b6 && (!isfirst)) {
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
                    (iv31[lid] - 1)];
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
      if (!b5) {
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
 *     ( v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
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
 * function [alpha_vs,us] = rescale_displacements(xs, us, tris, tol, alpha_in)
 */
static void b_rescale_displacements(const emxArray_real_T *xs, const
  emxArray_real_T *us, const emxArray_int32_T *tris, emxArray_real_T *alpha_vs)
{
  int32_T nv;
  int32_T ntri;
  int32_T i18;
  int32_T ii;
  real_T b_us[9];
  boolean_T x[9];
  boolean_T y;
  boolean_T exitg1;
  real_T b_xs[9];
  real_T alpha_tri;
  real_T u0;
  real_T minval[3];

  /* RESCALE_DISPLACEMENTS */
  /*  Given layers of surfaces and the displacement of a particular layer, */
  /*  scale the vertex displacements to avoid folding */
  /* 'async_scale_disps_tri_cleanmesh:40' coder.inline('never') */
  /* 'async_scale_disps_tri_cleanmesh:41' nv   = int32(size(xs,1)); */
  nv = xs->size[0];

  /* 'async_scale_disps_tri_cleanmesh:42' ntri = int32(size(tris,1)); */
  ntri = tris->size[0];

  /* 'async_scale_disps_tri_cleanmesh:43' alpha_vs = ones(nv,1); */
  i18 = alpha_vs->size[0];
  alpha_vs->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)alpha_vs, i18, (int32_T)sizeof(real_T));
  nv--;
  for (i18 = 0; i18 <= nv; i18++) {
    alpha_vs->data[i18] = 1.0;
  }

  /* 'async_scale_disps_tri_cleanmesh:45' for ii=1:ntri */
  for (ii = 0; ii + 1 <= ntri; ii++) {
    /* 'async_scale_disps_tri_cleanmesh:46' vs = tris(ii,1:3); */
    /* 'async_scale_disps_tri_cleanmesh:47' if nargin>6 && all(alpha_in(vs)==1) */
    /* 'async_scale_disps_tri_cleanmesh:52' us_tri = us(vs,1:3); */
    /* 'async_scale_disps_tri_cleanmesh:53' if all(us_tri(:)==0) */
    for (i18 = 0; i18 < 3; i18++) {
      for (nv = 0; nv < 3; nv++) {
        b_us[nv + 3 * i18] = us->data[(tris->data[ii + tris->size[0] * nv] +
          us->size[0] * i18) - 1];
      }
    }

    for (i18 = 0; i18 < 9; i18++) {
      x[i18] = (b_us[i18] == 0.0);
    }

    y = TRUE;
    nv = 1;
    exitg1 = 0U;
    while ((exitg1 == 0U) && (nv <= 9)) {
      if (x[nv - 1] == 0) {
        y = FALSE;
        exitg1 = 1U;
      } else {
        nv++;
      }
    }

    if (y) {
    } else {
      /* 'async_scale_disps_tri_cleanmesh:55' alpha_tri = check_prism( xs(vs,1:3), us_tri); */
      for (i18 = 0; i18 < 3; i18++) {
        for (nv = 0; nv < 3; nv++) {
          b_xs[nv + 3 * i18] = xs->data[(tris->data[ii + tris->size[0] * nv] +
            xs->size[0] * i18) - 1];
        }
      }

      for (i18 = 0; i18 < 3; i18++) {
        for (nv = 0; nv < 3; nv++) {
          b_us[nv + 3 * i18] = us->data[(tris->data[ii + tris->size[0] * nv] +
            us->size[0] * i18) - 1];
        }
      }

      alpha_tri = check_prism(b_xs, b_us);

      /* 'async_scale_disps_tri_cleanmesh:57' if alpha_tri < tol */
      if (alpha_tri < 0.1) {
        /* 'async_scale_disps_tri_cleanmesh:57' alpha_tri = 0.5*alpha_tri; */
        alpha_tri *= 0.5;
      }

      /* 'async_scale_disps_tri_cleanmesh:59' if alpha_tri<1 */
      if (alpha_tri < 1.0) {
        /* 'async_scale_disps_tri_cleanmesh:60' alpha_vs(vs) = min( alpha_vs(vs), alpha_tri); */
        for (nv = 0; nv < 3; nv++) {
          u0 = alpha_vs->data[tris->data[ii + tris->size[0] * nv] - 1];
          u0 = u0 <= alpha_tri ? u0 : alpha_tri;
          minval[nv] = u0;
        }

        for (i18 = 0; i18 < 3; i18++) {
          alpha_vs->data[tris->data[ii + tris->size[0] * i18] - 1] = minval[i18];
        }
      }
    }
  }
}

/*
 *
 */
static void b_sum(const real_T x[9], real_T y[3])
{
  int32_T ixstart;
  int32_T iy;
  int32_T i;
  real_T s;
  int32_T k;
  ixstart = -1;
  iy = -1;
  for (i = 0; i < 3; i++) {
    ixstart++;
    s = x[ixstart];
    for (k = 0; k < 2; k++) {
      ixstart++;
      s += x[ixstart];
    }

    iy++;
    y[iy] = s;
  }
}

/*
 * function bs = backsolve(R, bs, cend, ws)
 */
static void backsolve(const emxArray_real_T *R, emxArray_real_T *bs, int32_T
                      cend, const emxArray_real_T *ws)
{
  int32_T jj;
  int32_T ii;
  int32_T b_bs[2];
  int32_T c_bs[2];
  int32_T d_bs[2];
  emxArray_real_T e_bs;
  emxArray_real_T f_bs;
  emxArray_real_T g_bs;
  int32_T h_bs[2];
  int32_T i_bs[2];
  int32_T j_bs[2];
  int32_T k_bs[2];

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
  for (jj = cend - 1; jj + 1 > 0; jj--) {
    /* 'backsolve:14' for ii=jj+1:cend */
    for (ii = jj + 1; ii + 1 <= cend; ii++) {
      /* 'backsolve:15' bs(jj,kk) = bs(jj,kk) - R(jj,ii) * bs(ii,kk); */
      b_bs[0] = bs->size[0];
      b_bs[1] = 1;
      c_bs[0] = bs->size[0];
      c_bs[1] = 1;
      d_bs[0] = bs->size[0];
      d_bs[1] = 1;
      e_bs = *bs;
      e_bs.size = (int32_T *)&b_bs;
      e_bs.numDimensions = 1;
      f_bs = *bs;
      f_bs.size = (int32_T *)&c_bs;
      f_bs.numDimensions = 1;
      g_bs = *bs;
      g_bs.size = (int32_T *)&d_bs;
      g_bs.numDimensions = 1;
      e_bs.data[jj] = f_bs.data[jj] - R->data[jj + R->size[0] * ii] *
        g_bs.data[ii];
    }

    /* 'backsolve:18' assert( R(jj,jj)~=0); */
    /* 'backsolve:19' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
    h_bs[0] = bs->size[0];
    h_bs[1] = 1;
    i_bs[0] = bs->size[0];
    i_bs[1] = 1;
    e_bs = *bs;
    e_bs.size = (int32_T *)&h_bs;
    e_bs.numDimensions = 1;
    f_bs = *bs;
    f_bs.size = (int32_T *)&i_bs;
    f_bs.numDimensions = 1;
    e_bs.data[jj] = f_bs.data[jj] / R->data[jj + R->size[0] * jj];
  }

  /* 'backsolve:23' if nargin>3 */
  /*  Scale bs back if ts is given. */
  /* 'backsolve:25' for kk=1:int32(size(bs,2)) */
  /* 'backsolve:26' for jj = 1:cend */
  for (jj = 0; jj + 1 <= cend; jj++) {
    /* 'backsolve:27' bs(jj,kk) = bs(jj,kk) / ws(jj); */
    j_bs[0] = bs->size[0];
    j_bs[1] = 1;
    k_bs[0] = bs->size[0];
    k_bs[1] = 1;
    e_bs = *bs;
    e_bs.size = (int32_T *)&j_bs;
    e_bs.numDimensions = 1;
    f_bs = *bs;
    f_bs.size = (int32_T *)&k_bs;
    f_bs.numDimensions = 1;
    e_bs.data[jj] = f_bs.data[jj] / ws->data[jj];
  }
}

/*
 * function [bs,deg_out] = backsolve_bivar_safeguarded(R, bs, degree, interp, ws)
 */
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
  b_emxInit_real_T(&bs_bak, 1);

  /*  Perform back substitution with safeguards to downgrade the order if necessary. */
  /*      [bs,deg_out] = backsolve_bivar_safeguarded(R, bs, degree, interp, ws) */
  /* 'backsolve_bivar_safeguarded:5' coder.varsize( 'bs_bak', [28,1], [1,0]); */
  /* 'backsolve_bivar_safeguarded:6' coder.varsize( 'tb', [7,1], [1,0]); */
  /* 'backsolve_bivar_safeguarded:8' tol = 0.05; */
  /*  Second, solve for each degree in decending order */
  /* 'backsolve_bivar_safeguarded:11' if nargout>1 */
  /* 'backsolve_bivar_safeguarded:11' deg_out = coder.nullcopy( zeros(1,size(bs,2),'int32')); */
  /* 'backsolve_bivar_safeguarded:13' for kk=1:int32(size(bs,2)) */
  /* 'backsolve_bivar_safeguarded:14' deg = degree; */
  deg_out = degree;

  /* 'backsolve_bivar_safeguarded:15' ncols = int32(bitshift(uint32((deg+2)*(deg+1)),-1))-int32(interp); */
  ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);

  /* 'backsolve_bivar_safeguarded:16' bs_bak = coder.nullcopy( zeros(ncols,1)); */
  ii = bs_bak->size[0];
  bs_bak->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)bs_bak, ii, (int32_T)sizeof(real_T));

  /* 'backsolve_bivar_safeguarded:17' if deg>=2 */
  if (degree >= 2) {
    /*  Back up bs for computing reference solution and for resolving */
    /*  the backsolve after lowering degree. */
    /* 'backsolve_bivar_safeguarded:20' assert(ncols<=28); */
    /* 'backsolve_bivar_safeguarded:21' for i=1:ncols */
    for (i = 0; i + 1 <= ncols; i++) {
      /* 'backsolve_bivar_safeguarded:21' bs_bak(i) = bs(i,kk); */
      b_bs[0] = bs->size[0];
      b_bs[1] = 1;
      c_bs = *bs;
      c_bs.size = (int32_T *)&b_bs;
      c_bs.numDimensions = 1;
      bs_bak->data[i] = c_bs.data[i];
    }
  }

  /* 'backsolve_bivar_safeguarded:24' while deg>=1 */
  b_emxInit_real_T(&tb, 1);
  exitg1 = 0U;
  while ((exitg1 == 0U) && (deg_out >= 1)) {
    /* 'backsolve_bivar_safeguarded:25' cend = ncols; */
    cend = ncols;

    /* 'backsolve_bivar_safeguarded:26' downgrade = false; */
    downgrade = FALSE;

    /* 'backsolve_bivar_safeguarded:28' for d = deg:-1:int32(interp) */
    d = deg_out;
    exitg2 = 0U;
    while ((exitg2 == 0U) && (d > -1)) {
      /* 'backsolve_bivar_safeguarded:29' cstart = int32(bitshift(uint32(d*(d+1)),-1))+1-int32(interp); */
      cstart = (int32_T)((uint32_T)(d * (d + 1)) >> 1U);

      /*  Solve for bs */
      /* 'backsolve_bivar_safeguarded:32' for jj=cend:-1:cstart */
      for (i = cend - 1; i + 1 >= cstart + 1; i--) {
        /* 'backsolve_bivar_safeguarded:33' for ii=jj+1:ncols */
        for (ii = i + 1; ii + 1 <= ncols; ii++) {
          /* 'backsolve_bivar_safeguarded:34' bs(jj,kk) = bs(jj,kk) - R(jj,ii) * bs(ii,kk); */
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

        /* 'backsolve_bivar_safeguarded:36' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
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
      /* 'backsolve_bivar_safeguarded:41' if d>=2 && d<deg */
      guard1 = FALSE;
      if ((d >= 2) && (d < deg_out)) {
        /* 'backsolve_bivar_safeguarded:42' tb = bs_bak(cstart:cend); */
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

        /* 'backsolve_bivar_safeguarded:44' for jj=cend:-1:cstart */
        i = cend - 1;
        exitg3 = 0U;
        while ((exitg3 == 0U) && (i + 1 >= cstart + 1)) {
          /* 'backsolve_bivar_safeguarded:45' jind = jj-cstart+1; */
          jind = i - cstart;

          /* 'backsolve_bivar_safeguarded:46' for ii=jj+1:cend */
          for (ii = i + 2; ii <= cend; ii++) {
            /* 'backsolve_bivar_safeguarded:47' tb(jind) = tb(jind) - R(jj,ii) * tb(ii-cstart+1); */
            tb->data[jind] -= R->data[i + R->size[0] * (ii - 1)] * tb->data[(ii
              - cstart) - 1];
          }

          /* 'backsolve_bivar_safeguarded:49' tb(jind)  = tb(jind) / R(jj,jj); */
          tb->data[jind] /= R->data[i + R->size[0] * i];

          /* 'backsolve_bivar_safeguarded:51' err = abs(bs(jj,kk)-tb(jind)); */
          k_bs[0] = bs->size[0];
          k_bs[1] = 1;
          c_bs = *bs;
          c_bs.size = (int32_T *)&k_bs;
          c_bs.numDimensions = 1;
          err = c_bs.data[i] - tb->data[jind];
          err = fabs(err);

          /* 'backsolve_bivar_safeguarded:52' if err > tol && err >= (1+tol)*abs(tb(jind)) */
          if ((err > 0.05) && (err >= 1.05 * fabs(tb->data[jind]))) {
            /* 'backsolve_bivar_safeguarded:53' downgrade = true; */
            downgrade = TRUE;
            exitg3 = 1U;
          } else {
            i--;
          }
        }

        /* 'backsolve_bivar_safeguarded:58' if downgrade */
        if (downgrade) {
          exitg2 = 1U;
        } else {
          guard1 = TRUE;
        }
      } else {
        guard1 = TRUE;
      }

      if (guard1 == TRUE) {
        /* 'backsolve_bivar_safeguarded:61' cend = cstart - 1; */
        cend = cstart;
        d--;
      }
    }

    /* 'backsolve_bivar_safeguarded:64' if ~downgrade */
    if (!downgrade) {
      exitg1 = 1U;
    } else {
      /* 'backsolve_bivar_safeguarded:66' else */
      /*  Decrease the degree of fitting by one. */
      /*  An alternative is to decreaes deg to d. This may be more */
      /*  efficient but it lose some chances to obtain higher-order accuracy. */
      /* 'backsolve_bivar_safeguarded:70' deg = deg - 1; */
      deg_out--;

      /* 'backsolve_bivar_safeguarded:71' ncols = int32(bitshift(uint32((deg+2)*(deg+1)),-1))-int32(interp); */
      ncols = (int32_T)((uint32_T)((deg_out + 2) * (deg_out + 1)) >> 1U);

      /*  Restore bs. */
      /* 'backsolve_bivar_safeguarded:73' bs(1:ncols,kk) = bs_bak(1:ncols); */
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
  /* 'backsolve_bivar_safeguarded:78' if nargout>1 */
  /* 'backsolve_bivar_safeguarded:78' deg_out(kk) = deg; */
  /* 'backsolve_bivar_safeguarded:80' if nargin>4 */
  /*  Scale back bs. */
  /* 'backsolve_bivar_safeguarded:82' for jj = 1:ncols */
  for (i = 0; i + 1 <= ncols; i++) {
    /* 'backsolve_bivar_safeguarded:82' bs(jj,kk) = bs(jj,kk) / ws(jj); */
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

  /* 'backsolve_bivar_safeguarded:84' for jj = ncols+1:int32(size(bs,1)) */
  i = bs->size[0];
  while (ncols + 1 <= i) {
    /* 'backsolve_bivar_safeguarded:84' bs(jj,kk) = 0; */
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

/*
 * function [us_smooth, nrms] = adjust_disps_onto_hisurf_cleanmesh( nv_clean, ps, us_smooth, nrms, ...
 *     tris, opphes, isridge, ridgeedge, flabel, args)
 */
static void c_adjust_disps_onto_hisurf_clea(int32_T nv_clean, const
  emxArray_real_T *ps, emxArray_real_T *us_smooth, const emxArray_real_T *nrms,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const char_T
  args_method[3], int32_T args_degree)
{
  int32_T method;
  emxArray_int32_T *v2he;
  emxArray_int32_T *allv_ngbvs;
  int32_T offset;
  int32_T nelems;
  emxArray_int32_T *allv_offsets;
  int32_T i13;
  int32_T i;
  emxArray_int32_T *r16;
  int32_T i14;
  emxArray_int32_T *r17;
  emxArray_int32_T *ngbvs1;
  emxArray_int32_T *ngbvs2;
  emxArray_int32_T *ngbvs3;
  emxArray_real_T *b_ps;
  emxArray_real_T *b_nrms;
  emxArray_real_T *c_ps;
  emxArray_real_T *c_nrms;
  emxArray_real_T *d_ps;
  emxArray_real_T *d_nrms;
  emxArray_real_T *e_ps;
  emxArray_real_T *e_nrms;
  emxArray_real_T *f_ps;
  emxArray_real_T *f_nrms;
  emxArray_real_T *g_ps;
  emxArray_real_T *g_nrms;
  emxArray_real_T *h_ps;
  emxArray_real_T *h_nrms;
  emxArray_real_T *i_ps;
  emxArray_real_T *i_nrms;
  emxArray_real_T *j_ps;
  emxArray_real_T *j_nrms;
  emxArray_real_T *k_ps;
  emxArray_real_T *k_nrms;
  emxArray_real_T *l_ps;
  emxArray_real_T *m_ps;
  emxArray_real_T *n_ps;
  emxArray_real_T *o_ps;
  emxArray_real_T *p_ps;
  real_T pnt1[3];
  real_T xi;
  int8_T loc;
  real_T lcoor[2];
  int32_T fid;
  real_T d4;
  real_T pos[3];
  real_T pnt2[3];
  real_T pnt3[3];
  int8_T verts[2];

  /*  Adjust the given displacements so that the updated displacements would */
  /*  be on the high-order reconstruction of a surface mesh. */
  /* 'adjust_disps_onto_hisurf_cleanmesh:11' coder.extrinsic('fprintf'); */
  /* 'adjust_disps_onto_hisurf_cleanmesh:12' coder.inline('never') */
  /* 'adjust_disps_onto_hisurf_cleanmesh:13' WALF=1; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:14' WALF_NRM=2; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:15' CMF=3; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:16' CMF_NRM=4; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:18' if strcmp(args.method,'walf') || strcmp(args.method,'WALF') */
  if (eml_strcmp(args_method) || b_eml_strcmp(args_method)) {
    /* 'adjust_disps_onto_hisurf_cleanmesh:19' method = WALF; */
    method = 1;
  } else if (c_eml_strcmp(args_method) || d_eml_strcmp(args_method)) {
    /* 'adjust_disps_onto_hisurf_cleanmesh:20' elseif strcmp(args.method, 'walf_nrm') || strcmp(args.method, 'WALF_NRM') */
    /* 'adjust_disps_onto_hisurf_cleanmesh:21' method = WALF_NRM; */
    method = 2;
  } else if (e_eml_strcmp(args_method) || f_eml_strcmp(args_method)) {
    /* 'adjust_disps_onto_hisurf_cleanmesh:22' elseif strcmp(args.method, 'cmf') || strcmp(args.method, 'CMF') */
    /* 'adjust_disps_onto_hisurf_cleanmesh:23' method = CMF; */
    method = 3;
  } else if (g_eml_strcmp(args_method) || h_eml_strcmp(args_method)) {
    /* 'adjust_disps_onto_hisurf_cleanmesh:24' elseif strcmp(args.method, 'cmf_nrm') || strcmp(args.method, 'CMF_NRM') */
    /* 'adjust_disps_onto_hisurf_cleanmesh:25' method = CMF_NRM; */
    method = 4;
  } else {
    /* 'adjust_disps_onto_hisurf_cleanmesh:26' else */
    /* 'adjust_disps_onto_hisurf_cleanmesh:27' error('Unknown method %s', args.method); */
    /* 'adjust_disps_onto_hisurf_cleanmesh:28' method = 0; */
    method = 0;
  }

  emxInit_int32_T(&v2he, 1);
  emxInit_int32_T(&allv_ngbvs, 1);

  /* % Collect neighbors for all vertices */
  /* 'adjust_disps_onto_hisurf_cleanmesh:32' tol_dist = 1.e-6; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:33' ring = double(args.degree+1)*0.5; */
  /* 'adjust_disps_onto_hisurf_cleanmesh:34' v2he = determine_incident_halfedges( tris, opphes); */
  c_determine_incident_halfedges(tris, opphes, v2he);

  /* 'adjust_disps_onto_hisurf_cleanmesh:35' allv_ngbvs = obtain_all_nring_surf( ring, int32(size(ps,1)), tris, opphes, v2he); */
  obtain_all_nring_surf((real_T)(args_degree + 1) * 0.5, ps->size[0], tris,
                        opphes, v2he, allv_ngbvs);

  /* 'adjust_disps_onto_hisurf_cleanmesh:36' allv_offsets = determine_offsets_mixed_elems( allv_ngbvs); */
  /*  Determine the offsets of each element in a mixed connectivity table. */
  /*  */
  /*  OFFSETS = DETERMINE_OFFSETS_MIXED_ELEMS( ELEMS) */
  /*  */
  /*  At input, ELEMS is a column vector, with format */
  /*      [e1_nv, e1_v1,e1_v2,..., e2_nv, e2_v1,e2_v2, ...]. */
  /*  At output, OFFSETS contains the beginning position for each element */
  /*       in elems (i.e., the index of ei_nv for the ith element). */
  /*  */
  /*  Note that you can also use this function for the table of opposite */
  /*       half-faces (opphfs) instead of element connectivity. In this case, */
  /*       the first input argument should have format */
  /*       [e1_nf, e1_opphf1,e1_opphf2,..., e2_nf, e2_opphf1, e2_opphf2, ...]. */
  /*  */
  /*  See also LINEARIZE_MIXED_ELEMS, REGULARIZE_MIXED_ELEMS. */
  /* 'determine_offsets_mixed_elems:20' assert(size(elems,2)==1); */
  /*  Allocate memory space for elems and determine number of elems */
  /* 'determine_offsets_mixed_elems:23' offset=int32(1); */
  offset = 1;

  /* 'determine_offsets_mixed_elems:23' nelems=int32(0); */
  nelems = 0;

  /* 'determine_offsets_mixed_elems:24' while offset<size(elems,1) */
  while (offset < allv_ngbvs->size[0]) {
    /* 'determine_offsets_mixed_elems:26' nelems = nelems + 1; */
    nelems++;

    /* 'determine_offsets_mixed_elems:27' offset = offset+elems(offset)+1; */
    offset = (offset + allv_ngbvs->data[offset - 1]) + 1;
  }

  emxInit_int32_T(&allv_offsets, 1);

  /* 'determine_offsets_mixed_elems:30' offsets = coder.nullcopy(zeros(nelems,1,'int32')); */
  i13 = allv_offsets->size[0];
  allv_offsets->size[0] = nelems;
  emxEnsureCapacity((emxArray__common *)allv_offsets, i13, (int32_T)sizeof
                    (int32_T));

  /*  Set offset for the elements. */
  /* 'determine_offsets_mixed_elems:34' offset=int32(1); */
  offset = 1;

  /* 'determine_offsets_mixed_elems:34' i=int32(1); */
  i = 1;

  /* 'determine_offsets_mixed_elems:35' while offset<size(elems,1) && i<=nelems */
  while ((offset < allv_ngbvs->size[0]) && (i <= nelems)) {
    /* 'determine_offsets_mixed_elems:36' offsets(i) = offset; */
    allv_offsets->data[i - 1] = offset;

    /* 'determine_offsets_mixed_elems:37' offset = offset+elems(offset)+1; */
    offset = (offset + allv_ngbvs->data[offset - 1]) + 1;

    /* 'determine_offsets_mixed_elems:38' i = i + 1; */
    i++;
  }

  /*  Reset the rest to zero. */
  /* 'determine_offsets_mixed_elems:42' offsets(i:end) = 0; */
  i13 = allv_offsets->size[0];
  if (i > i13) {
    i = 1;
    i13 = 0;
  }

  emxInit_int32_T(&r16, 1);
  i14 = r16->size[0];
  r16->size[0] = (i13 - i) + 1;
  emxEnsureCapacity((emxArray__common *)r16, i14, (int32_T)sizeof(int32_T));
  nelems = i13 - i;
  for (i13 = 0; i13 <= nelems; i13++) {
    r16->data[i13] = i + i13;
  }

  b_emxInit_int32_T(&r17, 2);
  i13 = r17->size[0] * r17->size[1];
  r17->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r17, i13, (int32_T)sizeof(int32_T));
  offset = r16->size[0];
  i13 = r17->size[0] * r17->size[1];
  r17->size[1] = offset;
  emxEnsureCapacity((emxArray__common *)r17, i13, (int32_T)sizeof(int32_T));
  nelems = r16->size[0] - 1;
  for (i13 = 0; i13 <= nelems; i13++) {
    r17->data[i13] = r16->data[i13];
  }

  emxFree_int32_T(&r16);
  nelems = r17->size[0] * r17->size[1] - 1;
  for (i13 = 0; i13 <= nelems; i13++) {
    allv_offsets->data[r17->data[i13] - 1] = 0;
  }

  emxFree_int32_T(&r17);

  /*  Loop through the points to project them onto the surface one by one. */
  /* 'adjust_disps_onto_hisurf_cleanmesh:39' for i= 1: nv_clean */
  i = 0;
  emxInit_int32_T(&ngbvs1, 1);
  emxInit_int32_T(&ngbvs2, 1);
  emxInit_int32_T(&ngbvs3, 1);
  emxInit_real_T(&b_ps, 2);
  emxInit_real_T(&b_nrms, 2);
  emxInit_real_T(&c_ps, 2);
  emxInit_real_T(&c_nrms, 2);
  emxInit_real_T(&d_ps, 2);
  emxInit_real_T(&d_nrms, 2);
  emxInit_real_T(&e_ps, 2);
  emxInit_real_T(&e_nrms, 2);
  emxInit_real_T(&f_ps, 2);
  emxInit_real_T(&f_nrms, 2);
  emxInit_real_T(&g_ps, 2);
  emxInit_real_T(&g_nrms, 2);
  emxInit_real_T(&h_ps, 2);
  emxInit_real_T(&h_nrms, 2);
  emxInit_real_T(&i_ps, 2);
  emxInit_real_T(&i_nrms, 2);
  emxInit_real_T(&j_ps, 2);
  emxInit_real_T(&j_nrms, 2);
  emxInit_real_T(&k_ps, 2);
  emxInit_real_T(&k_nrms, 2);
  emxInit_real_T(&l_ps, 2);
  emxInit_real_T(&m_ps, 2);
  emxInit_real_T(&n_ps, 2);
  emxInit_real_T(&o_ps, 2);
  emxInit_real_T(&p_ps, 2);
  while (i + 1 <= nv_clean) {
    /* 'adjust_disps_onto_hisurf_cleanmesh:40' pnt = (ps(i,1:3)+us_smooth(i,1:3))'; */
    for (i13 = 0; i13 < 3; i13++) {
      pnt1[i13] = ps->data[i + ps->size[0] * i13] + us_smooth->data[i +
        us_smooth->size[0] * i13];
    }

    /* 'adjust_disps_onto_hisurf_cleanmesh:41' heid = v2he(i); */
    /*  Determine the triangle onto which the point should project */
    /* 'adjust_disps_onto_hisurf_cleanmesh:44' [fid, lcoor, loc, dist, proj] = find_parent_triangle(pnt, heid, ps, nrms, tris, opphes, v2he); */
    find_parent_triangle(pnt1, v2he->data[i], ps, nrms, tris, opphes, v2he, &fid,
                         lcoor, &loc, &xi, &offset);

    /* 'adjust_disps_onto_hisurf_cleanmesh:45' if (proj == 1) */
    /* 'adjust_disps_onto_hisurf_cleanmesh:49' if loc==0 || dist>tol_dist */
    if ((loc == 0) || (xi > 1.0E-6)) {
      /* 'adjust_disps_onto_hisurf_cleanmesh:50' offset1 = allv_offsets(tris(fid,1)); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:51' ngbvs1 = [tris(fid,1); allv_ngbvs( offset1+1:offset1+allv_ngbvs( offset1))]; */
      i13 = allv_offsets->data[tris->data[fid - 1] - 1] + 1;
      i14 = allv_offsets->data[tris->data[fid - 1] - 1] + allv_ngbvs->
        data[allv_offsets->data[tris->data[fid - 1] - 1] - 1];
      if (i13 > i14) {
        i13 = 1;
        i14 = 0;
      }

      offset = ngbvs1->size[0];
      ngbvs1->size[0] = (i14 - i13) + 2;
      emxEnsureCapacity((emxArray__common *)ngbvs1, offset, (int32_T)sizeof
                        (int32_T));
      ngbvs1->data[0] = tris->data[fid - 1];
      nelems = i14 - i13;
      for (i14 = 0; i14 <= nelems; i14++) {
        ngbvs1->data[i14 + 1] = allv_ngbvs->data[(i13 + i14) - 1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:52' ngbpnts1 = ps(ngbvs1,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:52' nrms1 = nrms(ngbvs1,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:54' offset2 = allv_offsets(tris(fid,2)); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:55' ngbvs2 = [tris(fid,2); allv_ngbvs( offset2+1:offset2+allv_ngbvs( offset2))]; */
      i13 = allv_offsets->data[tris->data[(fid + tris->size[0]) - 1] - 1] + 1;
      i14 = allv_offsets->data[tris->data[(fid + tris->size[0]) - 1] - 1] +
        allv_ngbvs->data[allv_offsets->data[tris->data[(fid + tris->size[0]) - 1]
        - 1] - 1];
      if (i13 > i14) {
        i13 = 1;
        i14 = 0;
      }

      offset = ngbvs2->size[0];
      ngbvs2->size[0] = (i14 - i13) + 2;
      emxEnsureCapacity((emxArray__common *)ngbvs2, offset, (int32_T)sizeof
                        (int32_T));
      ngbvs2->data[0] = tris->data[(fid + tris->size[0]) - 1];
      nelems = i14 - i13;
      for (i14 = 0; i14 <= nelems; i14++) {
        ngbvs2->data[i14 + 1] = allv_ngbvs->data[(i13 + i14) - 1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:56' ngbpnts2 = ps(ngbvs2,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:56' nrms2 = nrms(ngbvs2,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:58' offset3 = allv_offsets(tris(fid,1)); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:59' ngbvs3 = [tris(fid,3); allv_ngbvs( offset3+1:offset3+allv_ngbvs( offset3))]; */
      i13 = allv_offsets->data[tris->data[fid - 1] - 1] + 1;
      i14 = allv_offsets->data[tris->data[fid - 1] - 1] + allv_ngbvs->
        data[allv_offsets->data[tris->data[fid - 1] - 1] - 1];
      if (i13 > i14) {
        i13 = 1;
        i14 = 0;
      }

      offset = ngbvs3->size[0];
      ngbvs3->size[0] = (i14 - i13) + 2;
      emxEnsureCapacity((emxArray__common *)ngbvs3, offset, (int32_T)sizeof
                        (int32_T));
      ngbvs3->data[0] = tris->data[(fid + (tris->size[0] << 1)) - 1];
      nelems = i14 - i13;
      for (i14 = 0; i14 <= nelems; i14++) {
        ngbvs3->data[i14 + 1] = allv_ngbvs->data[(i13 + i14) - 1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:60' ngbpnts3 = ps(ngbvs3,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:60' nrms3 = nrms(ngbvs3,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:62' switch method */
      switch (method) {
       case 1:
        /* 'adjust_disps_onto_hisurf_cleanmesh:63' case WALF */
        /* 'adjust_disps_onto_hisurf_cleanmesh:64' pnt = polyfit3d_walf_tri(ngbpnts1, nrms1, ngbpnts2, nrms2, ... */
        /* 'adjust_disps_onto_hisurf_cleanmesh:65'                     ngbpnts3, nrms3, lcoor(1), lcoor(2), args.degree); */
        offset = args_degree;

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
        /* 'polyfit3d_walf_tri:22' if nargin<9 || deg==0 */
        if (args_degree == 0) {
          /* 'polyfit3d_walf_tri:22' deg = int32(2); */
          offset = 2;
        }

        /*  Do not use interpolation by default */
        /* 'polyfit3d_walf_tri:25' if nargin<10 */
        /* 'polyfit3d_walf_tri:25' interp = false; */
        /*  Compute face normal vector and the local coordinate */
        /*  np = int32(size(xi,1)); */
        /*  pos = coder.nullcopy(zeros(np,3)); */
        /*  for i = 1:np */
        /*      pos(i,:) = (1-xi(i)-eta(i)).*ngbpnts1(1,1:3) + xi(i).*ngbpnts2(1,1:3) + eta(i).*ngbpnts3(1,1:3); */
        /*  end */
        /* 'polyfit3d_walf_tri:33' pos = (1-xi-eta).*ngbpnts1(1,1:3) +xi*ngbpnts2(1,1:3)+eta*ngbpnts3(1,1:3); */
        d4 = (1.0 - lcoor[0]) - lcoor[1];
        i13 = n_ps->size[0] * n_ps->size[1];
        n_ps->size[0] = ngbvs1->size[0];
        n_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)n_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            n_ps->data[i14 + n_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = o_ps->size[0] * o_ps->size[1];
        o_ps->size[0] = ngbvs2->size[0];
        o_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)o_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            o_ps->data[i14 + o_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = p_ps->size[0] * p_ps->size[1];
        p_ps->size[0] = ngbvs3->size[0];
        p_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)p_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs3->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            p_ps->data[i14 + p_ps->size[0] * i13] = ps->data[(ngbvs3->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        for (i13 = 0; i13 < 3; i13++) {
          pos[i13] = (d4 * n_ps->data[n_ps->size[0] * i13] + lcoor[0] *
                      o_ps->data[o_ps->size[0] * i13]) + lcoor[1] * p_ps->
            data[p_ps->size[0] * i13];
        }

        /*  Interpolate using vertex-based polynomial fittings at three vertices */
        /* 'polyfit3d_walf_tri:36' pnt1 = polyfit3d_walf_vertex(ngbpnts1, nrms1, pos, deg, interp); */
        i13 = h_ps->size[0] * h_ps->size[1];
        h_ps->size[0] = ngbvs1->size[0];
        h_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)h_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            h_ps->data[i14 + h_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = h_nrms->size[0] * h_nrms->size[1];
        h_nrms->size[0] = ngbvs1->size[0];
        h_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)h_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            h_nrms->data[i14 + h_nrms->size[0] * i13] = nrms->data[(ngbvs1->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_walf_vertex(h_ps, h_nrms, pos, offset, pnt1);

        /* 'polyfit3d_walf_tri:37' pnt2 = polyfit3d_walf_vertex(ngbpnts2, nrms2, pos, deg, interp); */
        i13 = g_ps->size[0] * g_ps->size[1];
        g_ps->size[0] = ngbvs2->size[0];
        g_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)g_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            g_ps->data[i14 + g_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = g_nrms->size[0] * g_nrms->size[1];
        g_nrms->size[0] = ngbvs2->size[0];
        g_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)g_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            g_nrms->data[i14 + g_nrms->size[0] * i13] = nrms->data[(ngbvs2->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_walf_vertex(g_ps, g_nrms, pos, offset, pnt2);

        /* 'polyfit3d_walf_tri:38' pnt3 = polyfit3d_walf_vertex(ngbpnts3, nrms3, pos, deg, interp); */
        i13 = f_ps->size[0] * f_ps->size[1];
        f_ps->size[0] = ngbvs3->size[0];
        f_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)f_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs3->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            f_ps->data[i14 + f_ps->size[0] * i13] = ps->data[(ngbvs3->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = f_nrms->size[0] * f_nrms->size[1];
        f_nrms->size[0] = ngbvs3->size[0];
        f_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)f_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs3->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            f_nrms->data[i14 + f_nrms->size[0] * i13] = nrms->data[(ngbvs3->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_walf_vertex(f_ps, f_nrms, pos, offset, pnt3);

        /*  Compute weighted average of the three points */
        /* 'polyfit3d_walf_tri:41' pnt = (1-xi-eta).*pnt1 + xi.*pnt2 + eta.*pnt3; */
        d4 = (1.0 - lcoor[0]) - lcoor[1];
        for (i13 = 0; i13 < 3; i13++) {
          pnt1[i13] = (d4 * pnt1[i13] + lcoor[0] * pnt2[i13]) + lcoor[1] *
            pnt3[i13];
        }

        /*  pnt = coder.nullcopy(zeros(np,3)); */
        /*  for i = 1:np */
        /*      pnt(i,:) = (1-xi(i)-eta(i)).*pnt1(i,:) + xi(i).*pnt2(i,:) + eta(i).*pnt3(i,:);  */
        /*  end */
        break;

       case 3:
        /* 'adjust_disps_onto_hisurf_cleanmesh:66' case CMF */
        /* 'adjust_disps_onto_hisurf_cleanmesh:67' pnt = polyfit3d_cmf_tri(ngbpnts1, nrms1, ngbpnts2, nrms2, ... */
        /* 'adjust_disps_onto_hisurf_cleanmesh:68'                     ngbpnts3, nrms3, lcoor(1), lcoor(2), args.degree); */
        i13 = i_ps->size[0] * i_ps->size[1];
        i_ps->size[0] = ngbvs1->size[0];
        i_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)i_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            i_ps->data[i14 + i_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = i_nrms->size[0] * i_nrms->size[1];
        i_nrms->size[0] = ngbvs1->size[0];
        i_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)i_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            i_nrms->data[i14 + i_nrms->size[0] * i13] = nrms->data[(ngbvs1->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        i13 = j_ps->size[0] * j_ps->size[1];
        j_ps->size[0] = ngbvs2->size[0];
        j_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)j_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            j_ps->data[i14 + j_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = j_nrms->size[0] * j_nrms->size[1];
        j_nrms->size[0] = ngbvs2->size[0];
        j_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)j_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            j_nrms->data[i14 + j_nrms->size[0] * i13] = nrms->data[(ngbvs2->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        i13 = k_ps->size[0] * k_ps->size[1];
        k_ps->size[0] = ngbvs3->size[0];
        k_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)k_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs3->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            k_ps->data[i14 + k_ps->size[0] * i13] = ps->data[(ngbvs3->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = k_nrms->size[0] * k_nrms->size[1];
        k_nrms->size[0] = ngbvs3->size[0];
        k_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)k_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs3->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            k_nrms->data[i14 + k_nrms->size[0] * i13] = nrms->data[(ngbvs3->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_cmf_tri(i_ps, i_nrms, j_ps, j_nrms, k_ps, k_nrms, lcoor[0],
                          lcoor[1], args_degree, pnt1);
        break;
      }
    } else if (loc >= 4) {
      /* 'adjust_disps_onto_hisurf_cleanmesh:70' elseif loc>=4 */
      /* 'adjust_disps_onto_hisurf_cleanmesh:71' pnt = ps(tris(fid,loc-3),1:3)'; */
      offset = tris->data[(fid + tris->size[0] * (loc - 4)) - 1];
      for (i13 = 0; i13 < 3; i13++) {
        pnt1[i13] = ps->data[(offset + ps->size[0] * i13) - 1];
      }

      /* pnt = ps(i,1:3)'; */
    } else {
      /* 'adjust_disps_onto_hisurf_cleanmesh:73' else */
      /* 'adjust_disps_onto_hisurf_cleanmesh:74' if loc==1 */
      if (loc == 1) {
        /* 'adjust_disps_onto_hisurf_cleanmesh:75' verts=[1; 2]; */
        for (i13 = 0; i13 < 2; i13++) {
          verts[i13] = (int8_T)(1 + i13);
        }

        /* 'adjust_disps_onto_hisurf_cleanmesh:75' xi = lcoor(1); */
        xi = lcoor[0];
      } else if (loc == 2) {
        /* 'adjust_disps_onto_hisurf_cleanmesh:76' elseif loc==2 */
        /* 'adjust_disps_onto_hisurf_cleanmesh:77' verts=[2; 3]; */
        for (i13 = 0; i13 < 2; i13++) {
          verts[i13] = (int8_T)(2 + i13);
        }

        /* 'adjust_disps_onto_hisurf_cleanmesh:77' xi = lcoor(2); */
        xi = lcoor[1];
      } else {
        /* 'adjust_disps_onto_hisurf_cleanmesh:78' else */
        /* 'adjust_disps_onto_hisurf_cleanmesh:79' assert( loc==3); */
        /* 'adjust_disps_onto_hisurf_cleanmesh:80' verts=[1; 3]; */
        for (i13 = 0; i13 < 2; i13++) {
          verts[i13] = (int8_T)(1 + (i13 << 1));
        }

        /* 'adjust_disps_onto_hisurf_cleanmesh:80' xi = lcoor(2); */
        xi = lcoor[1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:83' offset1 = allv_offsets(tris(fid,verts(1))); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:84' ngbvs1 = [tris(fid,verts(1)); allv_ngbvs( offset1+1:offset1+allv_ngbvs( offset1))]; */
      i13 = allv_offsets->data[tris->data[(fid + tris->size[0] * (verts[0] - 1))
        - 1] - 1] + 1;
      i14 = allv_offsets->data[tris->data[(fid + tris->size[0] * (verts[0] - 1))
        - 1] - 1] + allv_ngbvs->data[allv_offsets->data[tris->data[(fid +
        tris->size[0] * (verts[0] - 1)) - 1] - 1] - 1];
      if (i13 > i14) {
        i13 = 1;
        i14 = 0;
      }

      offset = ngbvs1->size[0];
      ngbvs1->size[0] = (i14 - i13) + 2;
      emxEnsureCapacity((emxArray__common *)ngbvs1, offset, (int32_T)sizeof
                        (int32_T));
      ngbvs1->data[0] = tris->data[(fid + tris->size[0] * (verts[0] - 1)) - 1];
      nelems = i14 - i13;
      for (i14 = 0; i14 <= nelems; i14++) {
        ngbvs1->data[i14 + 1] = allv_ngbvs->data[(i13 + i14) - 1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:85' ngbpnts1 = ps(ngbvs1,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:85' nrms1 = nrms(ngbvs1,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:87' offset2 = allv_offsets(tris(fid,verts(2))); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:88' ngbvs2 = [tris(fid,verts(2)); allv_ngbvs( offset2+1:offset2+allv_ngbvs( offset2))]; */
      i13 = allv_offsets->data[tris->data[(fid + tris->size[0] * (verts[1] - 1))
        - 1] - 1] + 1;
      i14 = allv_offsets->data[tris->data[(fid + tris->size[0] * (verts[1] - 1))
        - 1] - 1] + allv_ngbvs->data[allv_offsets->data[tris->data[(fid +
        tris->size[0] * (verts[1] - 1)) - 1] - 1] - 1];
      if (i13 > i14) {
        i13 = 1;
        i14 = 0;
      }

      offset = ngbvs2->size[0];
      ngbvs2->size[0] = (i14 - i13) + 2;
      emxEnsureCapacity((emxArray__common *)ngbvs2, offset, (int32_T)sizeof
                        (int32_T));
      ngbvs2->data[0] = tris->data[(fid + tris->size[0] * (verts[1] - 1)) - 1];
      nelems = i14 - i13;
      for (i14 = 0; i14 <= nelems; i14++) {
        ngbvs2->data[i14 + 1] = allv_ngbvs->data[(i13 + i14) - 1];
      }

      /* 'adjust_disps_onto_hisurf_cleanmesh:89' ngbpnts2 = ps(ngbvs2,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:89' nrms2 = nrms(ngbvs2,:); */
      /* 'adjust_disps_onto_hisurf_cleanmesh:91' switch method */
      switch (method) {
       case 1:
        /* 'adjust_disps_onto_hisurf_cleanmesh:92' case WALF */
        /* 'adjust_disps_onto_hisurf_cleanmesh:93' pnt = polyfit3d_walf_edge(ngbpnts1, nrms1, ngbpnts2, nrms2, ... */
        /* 'adjust_disps_onto_hisurf_cleanmesh:94'                     xi, args.degree); */
        offset = args_degree;

        /*  Compute the position of a point within an edge using */
        /*             weighted averaging of least-squares fittings. */
        /*  */
        /*  Input: */
        /*  ngbpnts1-2:Input points of size mx3, Its first column is x-coordinates, */
        /*             and its second column is y-coordinates. The first vertex will */
        /*             be used as the origin of the local coordinate system. */
        /*  nrms1-2:   The normals at ngbptns */
        /*  xi:        The parameter within the tangent line of edge */
        /*  deg:       The degree of polynomial to fit, from 1 to 6 */
        /*  interp:    If true, the fit is interpolatory at vertices. */
        /*  */
        /*  Output: */
        /*  pnt:       The reconstructed point in the global coordinate system */
        /*  */
        /*  See also polyfit3d_walf_tri, polyfit3d_walf_quad, polyfit3d_cmf_edge */
        /*  Use quadratic fitting by default */
        /* 'polyfit3d_walf_edge:21' if nargin<6 || deg==0 */
        if (args_degree == 0) {
          /* 'polyfit3d_walf_edge:21' deg = int32(2); */
          offset = 2;
        }

        /*  Do not use interpolation by default */
        /* 'polyfit3d_walf_edge:24' if nargin<7 */
        /* 'polyfit3d_walf_edge:24' interp = false; */
        /*  Compute face normal vector and the local coordinate */
        /* 'polyfit3d_walf_edge:27' pos = (1-xi).*ngbpnts1(1,1:3) + xi*ngbpnts2(1,1:3); */
        d4 = 1.0 - xi;
        i13 = l_ps->size[0] * l_ps->size[1];
        l_ps->size[0] = ngbvs1->size[0];
        l_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)l_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            l_ps->data[i14 + l_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = m_ps->size[0] * m_ps->size[1];
        m_ps->size[0] = ngbvs2->size[0];
        m_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)m_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            m_ps->data[i14 + m_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        for (i13 = 0; i13 < 3; i13++) {
          pos[i13] = d4 * l_ps->data[l_ps->size[0] * i13] + xi * m_ps->data
            [m_ps->size[0] * i13];
        }

        /*  Interpolate using vertex-based polynomial fittings at two vertices */
        /* 'polyfit3d_walf_edge:30' pnt1 = polyfit3d_walf_vertex(ngbpnts1, nrms1, pos, deg, interp); */
        i13 = c_ps->size[0] * c_ps->size[1];
        c_ps->size[0] = ngbvs1->size[0];
        c_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)c_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            c_ps->data[i14 + c_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = c_nrms->size[0] * c_nrms->size[1];
        c_nrms->size[0] = ngbvs1->size[0];
        c_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)c_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            c_nrms->data[i14 + c_nrms->size[0] * i13] = nrms->data[(ngbvs1->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_walf_vertex(c_ps, c_nrms, pos, offset, pnt1);

        /* 'polyfit3d_walf_edge:31' pnt2 = polyfit3d_walf_vertex(ngbpnts2, nrms2, pos, deg, interp); */
        i13 = b_ps->size[0] * b_ps->size[1];
        b_ps->size[0] = ngbvs2->size[0];
        b_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)b_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            b_ps->data[i14 + b_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = b_nrms->size[0] * b_nrms->size[1];
        b_nrms->size[0] = ngbvs2->size[0];
        b_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)b_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            b_nrms->data[i14 + b_nrms->size[0] * i13] = nrms->data[(ngbvs2->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_walf_vertex(b_ps, b_nrms, pos, offset, pnt2);

        /*  Compute weighted average of the two points */
        /* 'polyfit3d_walf_edge:34' pnt = (1-xi).*pnt1 +xi*pnt2; */
        d4 = 1.0 - xi;
        for (i13 = 0; i13 < 3; i13++) {
          pnt1[i13] = d4 * pnt1[i13] + xi * pnt2[i13];
        }
        break;

       case 3:
        /* 'adjust_disps_onto_hisurf_cleanmesh:95' case CMF */
        /* 'adjust_disps_onto_hisurf_cleanmesh:96' pnt = polyfit3d_cmf_edge(ngbpnts1, nrms1, ngbpnts2, nrms2, ... */
        /* 'adjust_disps_onto_hisurf_cleanmesh:97'                     xi, args.degree); */
        i13 = d_ps->size[0] * d_ps->size[1];
        d_ps->size[0] = ngbvs1->size[0];
        d_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)d_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            d_ps->data[i14 + d_ps->size[0] * i13] = ps->data[(ngbvs1->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = d_nrms->size[0] * d_nrms->size[1];
        d_nrms->size[0] = ngbvs1->size[0];
        d_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)d_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs1->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            d_nrms->data[i14 + d_nrms->size[0] * i13] = nrms->data[(ngbvs1->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        i13 = e_ps->size[0] * e_ps->size[1];
        e_ps->size[0] = ngbvs2->size[0];
        e_ps->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)e_ps, i13, (int32_T)sizeof(real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            e_ps->data[i14 + e_ps->size[0] * i13] = ps->data[(ngbvs2->data[i14]
              + ps->size[0] * i13) - 1];
          }
        }

        i13 = e_nrms->size[0] * e_nrms->size[1];
        e_nrms->size[0] = ngbvs2->size[0];
        e_nrms->size[1] = 3;
        emxEnsureCapacity((emxArray__common *)e_nrms, i13, (int32_T)sizeof
                          (real_T));
        for (i13 = 0; i13 < 3; i13++) {
          nelems = ngbvs2->size[0] - 1;
          for (i14 = 0; i14 <= nelems; i14++) {
            e_nrms->data[i14 + e_nrms->size[0] * i13] = nrms->data[(ngbvs2->
              data[i14] + nrms->size[0] * i13) - 1];
          }
        }

        polyfit3d_cmf_edge(d_ps, d_nrms, e_ps, e_nrms, xi, args_degree, pnt1);
        break;
      }
    }

    /* 'adjust_disps_onto_hisurf_cleanmesh:101' us_smooth( i,1:3) = pnt' - ps(i,1:3); */
    for (i13 = 0; i13 < 3; i13++) {
      us_smooth->data[i + us_smooth->size[0] * i13] = pnt1[i13] - ps->data[i +
        ps->size[0] * i13];
    }

    i++;
  }

  emxFree_real_T(&p_ps);
  emxFree_real_T(&o_ps);
  emxFree_real_T(&n_ps);
  emxFree_real_T(&m_ps);
  emxFree_real_T(&l_ps);
  emxFree_real_T(&k_nrms);
  emxFree_real_T(&k_ps);
  emxFree_real_T(&j_nrms);
  emxFree_real_T(&j_ps);
  emxFree_real_T(&i_nrms);
  emxFree_real_T(&i_ps);
  emxFree_real_T(&h_nrms);
  emxFree_real_T(&h_ps);
  emxFree_real_T(&g_nrms);
  emxFree_real_T(&g_ps);
  emxFree_real_T(&f_nrms);
  emxFree_real_T(&f_ps);
  emxFree_real_T(&e_nrms);
  emxFree_real_T(&e_ps);
  emxFree_real_T(&d_nrms);
  emxFree_real_T(&d_ps);
  emxFree_real_T(&c_nrms);
  emxFree_real_T(&c_ps);
  emxFree_real_T(&b_nrms);
  emxFree_real_T(&b_ps);
  emxFree_int32_T(&ngbvs3);
  emxFree_int32_T(&ngbvs2);
  emxFree_int32_T(&ngbvs1);
  emxFree_int32_T(&allv_offsets);
  emxFree_int32_T(&allv_ngbvs);
  emxFree_int32_T(&v2he);
}

/*
 * function nrms = average_vertex_normal_tri_cleanmesh(nv_clean, xs, tris, flabel)
 */
static void c_average_vertex_normal_tri_cle(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *flabel, emxArray_real_T *nrms)
{
  int32_T ntris;
  int32_T nv;
  int32_T i1;
  int32_T ii;
  int32_T iy;
  real_T a[3];
  real_T b[3];
  real_T nrm[3];
  int32_T k;
  real_T y;

  /* AVERAGE_VERTEX_NORMAL_TRI_CLEANMESH Compute average vertex normal for */
  /* clean submesh. */
  /* # coder.typeof( int32(0), [inf,3], [1,0]), coder.typeof( double(0), [inf,1], [1,0])} */
  /* 'average_vertex_normal_tri_cleanmesh:6' coder.inline('never') */
  /* 'average_vertex_normal_tri_cleanmesh:7' ntris = int32(size(tris, 1)); */
  ntris = tris->size[0];

  /* 'average_vertex_normal_tri_cleanmesh:8' nv = int32(size(xs, 1)); */
  nv = xs->size[0];

  /* 'average_vertex_normal_tri_cleanmesh:9' if nargin<4 */
  /* 'average_vertex_normal_tri_cleanmesh:10' nrms = zeros( nv, 3); */
  i1 = nrms->size[0] * nrms->size[1];
  nrms->size[0] = nv;
  nrms->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)nrms, i1, (int32_T)sizeof(real_T));
  nv = nv * 3 - 1;
  for (i1 = 0; i1 <= nv; i1++) {
    nrms->data[i1] = 0.0;
  }

  /* 'average_vertex_normal_tri_cleanmesh:11' for ii = 1 : ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'average_vertex_normal_tri_cleanmesh:12' if nargin>3 && flabel(ii) */
    if (flabel->data[ii] != 0.0) {
    } else {
      /* 'average_vertex_normal_tri_cleanmesh:13' nrm = cross_col( xs(tris(ii,3), 1:3)-xs(tris(ii,2), 1:3), ... */
      /* 'average_vertex_normal_tri_cleanmesh:14'         xs(tris(ii,1), 1:3)-xs(tris(ii,3), 1:3)); */
      nv = tris->data[ii + (tris->size[0] << 1)];
      iy = tris->data[ii + tris->size[0]];
      for (i1 = 0; i1 < 3; i1++) {
        a[i1] = xs->data[(nv + xs->size[0] * i1) - 1] - xs->data[(iy + xs->size
          [0] * i1) - 1];
      }

      nv = tris->data[ii];
      iy = tris->data[ii + (tris->size[0] << 1)];
      for (i1 = 0; i1 < 3; i1++) {
        b[i1] = xs->data[(nv + xs->size[0] * i1) - 1] - xs->data[(iy + xs->size
          [0] * i1) - 1];
      }

      /* CROSS_COL Efficient routine for computing cross product of two  */
      /* 3-dimensional column vectors. */
      /*  CROSS_COL(A,B) Efficiently computes the cross product between */
      /*  3-dimensional column vector A, and 3-dimensional column vector B. */
      /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
      nrm[0] = a[1] * b[2] - a[2] * b[1];
      nrm[1] = a[2] * b[0] - a[0] * b[2];
      nrm[2] = a[0] * b[1] - a[1] * b[0];

      /* 'average_vertex_normal_tri_cleanmesh:16' for jj = int32(1):3 */
      for (k = 0; k < 3; k++) {
        /* 'average_vertex_normal_tri_cleanmesh:17' nrms(tris(ii,jj), :) = nrms(tris(ii,jj), :) + nrm'; */
        nv = tris->data[ii + tris->size[0] * k];
        iy = tris->data[ii + tris->size[0] * k];
        for (i1 = 0; i1 < 3; i1++) {
          a[i1] = nrms->data[(iy + nrms->size[0] * i1) - 1] + nrm[i1];
        }

        for (i1 = 0; i1 < 3; i1++) {
          nrms->data[(nv + nrms->size[0] * i1) - 1] = a[i1];
        }
      }
    }
  }

  /* 'average_vertex_normal_tri_cleanmesh:21' for ii = 1:nv_clean */
  for (ii = 0; ii + 1 <= nv_clean; ii++) {
    /* 'average_vertex_normal_tri_cleanmesh:22' nrms(ii,:) = nrms(ii,:)/sqrt(nrms(ii,:)*nrms(ii,:)'+1.e-100); */
    for (i1 = 0; i1 < 3; i1++) {
      nrm[i1] = nrms->data[ii + nrms->size[0] * i1];
    }

    y = 0.0;
    nv = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrms->data[ii + nrms->size[0] * nv] * nrm[iy];
      nv++;
      iy++;
    }

    y = sqrt(y + 1.0E-100);
    for (i1 = 0; i1 < 3; i1++) {
      nrms->data[ii + nrms->size[0] * i1] /= y;
    }
  }
}

/*
 * function [min_angle, max_angle, min_area, max_area, min_valence, max_valence] = ...
 *     compute_statistics_tris_cleanmesh( nt_clean, xs, tris)
 */
static void c_compute_statistics_tris_clean(int32_T nt_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, real_T *min_angle, real_T
  *max_angle, real_T *min_area, real_T *max_area)
{
  real_T max_cos;
  real_T min_cos;
  int32_T kk;
  int32_T ix;
  int32_T iy;
  real_T b_xs[9];
  real_T c_xs[9];
  real_T d_xs[9];
  real_T e_xs[9];
  real_T f_xs[9];
  real_T g_xs[9];
  real_T ts_uv[9];
  real_T b_ts_uv[3];
  real_T c_ts_uv[3];
  real_T d_ts_uv[3];
  real_T e_ts_uv[3];
  real_T f_ts_uv[3];
  real_T g_ts_uv[3];
  real_T angles[3];
  real_T varargin_1[4];
  real_T nrm[3];
  real_T area;
  int32_T k;

  /*  compute minima and maxima of angles (in degrees), triangle areas, and */
  /*  vertex valences. */
  /* 'compute_statistics_tris_cleanmesh:7' coder.inline('never') */
  /* 'compute_statistics_tris_cleanmesh:8' max_cos = -1; */
  max_cos = -1.0;

  /* 'compute_statistics_tris_cleanmesh:8' min_cos =1; */
  min_cos = 1.0;

  /* 'compute_statistics_tris_cleanmesh:9' min_area = realmax; */
  *min_area = 1.7976931348623157E+308;

  /* 'compute_statistics_tris_cleanmesh:9' max_area = 0; */
  *max_area = 0.0;

  /* 'compute_statistics_tris_cleanmesh:10' ntri = int32(size(tris,1)); */
  /* 'compute_statistics_tris_cleanmesh:12' for kk=1:nt_clean */
  for (kk = 0; kk + 1 <= nt_clean; kk++) {
    /*      if (tris(kk,1)>nv_clean || tris(kk,2)>nv_clean || tris(kk,3)>nv_clean) */
    /*          continue; */
    /*      end */
    /* 'compute_statistics_tris_cleanmesh:16' xs_tri = xs( tris(kk,:), 1:3); */
    /* 'compute_statistics_tris_cleanmesh:17' ts_uv = [xs_tri(3,1:3)-xs_tri(2,1:3); xs_tri(1,1:3)-xs_tri(3,1:3); ... */
    /* 'compute_statistics_tris_cleanmesh:18'         xs_tri(2,1:3)-xs_tri(1,1:3)]; */
    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        b_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        c_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        d_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        e_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        f_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        g_xs[iy + 3 * ix] = xs->data[(tris->data[kk + tris->size[0] * iy] +
          xs->size[0] * ix) - 1];
      }
    }

    /* 'compute_statistics_tris_cleanmesh:20' angles = [cos_angle( -ts_uv(2,:)', ts_uv(3,:)'), ... */
    /* 'compute_statistics_tris_cleanmesh:21'         cos_angle( -ts_uv(3,:)', ts_uv(1,:)'), ... */
    /* 'compute_statistics_tris_cleanmesh:22'         cos_angle( -ts_uv(1,:)', ts_uv(2,:)')]; */
    for (ix = 0; ix < 3; ix++) {
      ts_uv[3 * ix] = b_xs[2 + 3 * ix] - c_xs[1 + 3 * ix];
      ts_uv[1 + 3 * ix] = d_xs[3 * ix] - e_xs[2 + 3 * ix];
      ts_uv[2 + 3 * ix] = f_xs[1 + 3 * ix] - g_xs[3 * ix];
      b_ts_uv[ix] = -ts_uv[1 + 3 * ix];
      c_ts_uv[ix] = ts_uv[2 + 3 * ix];
      d_ts_uv[ix] = -ts_uv[2 + 3 * ix];
      e_ts_uv[ix] = ts_uv[3 * ix];
      f_ts_uv[ix] = -ts_uv[3 * ix];
      g_ts_uv[ix] = ts_uv[1 + 3 * ix];
    }

    angles[0] = cos_angle(b_ts_uv, c_ts_uv);
    angles[1] = cos_angle(d_ts_uv, e_ts_uv);
    angles[2] = cos_angle(f_ts_uv, g_ts_uv);

    /* 'compute_statistics_tris_cleanmesh:24' max_cos = max([max_cos, angles]); */
    varargin_1[0] = max_cos;
    for (ix = 0; ix < 3; ix++) {
      varargin_1[ix + 1] = angles[ix];
    }

    max_cos = varargin_1[0];
    for (ix = 0; ix < 3; ix++) {
      if (varargin_1[ix + 1] > max_cos) {
        max_cos = varargin_1[ix + 1];
      }
    }

    /* 'compute_statistics_tris_cleanmesh:25' min_cos = min([min_cos, angles]); */
    varargin_1[0] = min_cos;
    for (ix = 0; ix < 3; ix++) {
      varargin_1[ix + 1] = angles[ix];
    }

    min_cos = varargin_1[0];
    for (ix = 0; ix < 3; ix++) {
      if (varargin_1[ix + 1] < min_cos) {
        min_cos = varargin_1[ix + 1];
      }
    }

    /* nrm = cross_col(ts_uv(1,:),ts_uv(2,:)); */
    /* 'compute_statistics_tris_cleanmesh:28' nrm = cross_col(ts_uv(1,:),ts_uv(3,:)); */
    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    nrm[0] = ts_uv[3] * ts_uv[8] - ts_uv[6] * ts_uv[5];
    nrm[1] = ts_uv[6] * ts_uv[2] - ts_uv[0] * ts_uv[8];
    nrm[2] = ts_uv[0] * ts_uv[5] - ts_uv[3] * ts_uv[2];

    /* 'compute_statistics_tris_cleanmesh:29' area = sqrt(nrm'*nrm); */
    area = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      area += nrm[ix] * nrm[iy];
      ix++;
      iy++;
    }

    area = sqrt(area);

    /* 'compute_statistics_tris_cleanmesh:30' min_area = min(area, min_area); */
    *min_area = area <= *min_area ? area : *min_area;

    /* 'compute_statistics_tris_cleanmesh:31' max_area = max(area, max_area); */
    *max_area = area >= *max_area ? area : *max_area;
  }

  /* 'compute_statistics_tris_cleanmesh:33' if max_cos>1 */
  if (max_cos > 1.0) {
    /* 'compute_statistics_tris_cleanmesh:33' max_cos=1; */
    max_cos = 1.0;
  }

  /* 'compute_statistics_tris_cleanmesh:34' if min_cos<-1 */
  if (min_cos < -1.0) {
    /* 'compute_statistics_tris_cleanmesh:34' min_cos=-1; */
    min_cos = -1.0;
  }

  /* 'compute_statistics_tris_cleanmesh:35' max_angle=acos(min_cos)/pi*180; */
  *max_angle = acos(min_cos) / 3.1415926535897931 * 180.0;

  /* 'compute_statistics_tris_cleanmesh:36' min_angle=acos(max_cos)/pi*180; */
  *min_angle = acos(max_cos) / 3.1415926535897931 * 180.0;

  /*  First, count the valence of each vertex. */
  /* 'compute_statistics_tris_cleanmesh:39' if nargout>4 */
}

/*
 * function us_smooth = constrained_smooth_surf_cleanmesh(nv_clean, xs, tris, ...
 *     flabels, isridge, grads_smooth, Hs_smooth, check_trank)
 */
static void c_constrained_smooth_surf_clean(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_int32_T
  *flabels, const emxArray_boolean_T *isridge, const emxArray_real_T
  *grads_smooth, const emxArray_real_T *Hs_smooth, boolean_T check_trank,
  emxArray_real_T *us_smooth)
{
  emxArray_real_T *Vs;
  emxArray_real_T *bs_m;
  emxArray_real_T *ns_constrained;
  emxArray_real_T *nrms_surf;
  emxArray_int8_T *tranks;
  emxArray_real_T *b_Vs;
  int32_T k;
  int32_T ix;
  emxArray_boolean_T *isfree;
  int32_T nv;
  int32_T jj;
  real_T y[3];
  real_T h;
  int32_T iy;
  real_T g;
  real_T nrm[3];
  real_T absnrm[3];
  real_T T[6];
  real_T b_absnrm[3];
  real_T b_T[6];
  real_T H2[4];
  real_T R2[4];
  real_T g2[2];
  boolean_T exitg1;
  boolean_T b_ns_constrained[3];
  c_emxInit_real_T(&Vs, 3);
  emxInit_real_T(&bs_m, 2);
  emxInit_real_T(&ns_constrained, 2);

  /* 'constrained_smooth_surf_cleanmesh:4' if nargin<7 */
  /* 'constrained_smooth_surf_cleanmesh:6' [Vs, bs_m, ns_constrained] = compute_medial_quadric_tri( xs, tris, flabels); */
  compute_medial_quadric_tri(xs, tris, flabels, Vs, bs_m, ns_constrained);

  /* 'constrained_smooth_surf_cleanmesh:8' if check_trank */
  emxInit_real_T(&nrms_surf, 2);
  emxInit_int8_T(&tranks, 1);
  if (check_trank) {
    c_emxInit_real_T(&b_Vs, 3);

    /* 'constrained_smooth_surf_cleanmesh:9' [nrms_surf, Vs, tranks] = eigenanalysis_surf( Vs, bs_m, isridge); */
    k = b_Vs->size[0] * b_Vs->size[1] * b_Vs->size[2];
    b_Vs->size[0] = 3;
    b_Vs->size[1] = 3;
    b_Vs->size[2] = Vs->size[2];
    emxEnsureCapacity((emxArray__common *)b_Vs, k, (int32_T)sizeof(real_T));
    ix = Vs->size[0] * Vs->size[1] * Vs->size[2] - 1;
    for (k = 0; k <= ix; k++) {
      b_Vs->data[k] = Vs->data[k];
    }

    eigenanalysis_surf(b_Vs, bs_m, isridge, nrms_surf, Vs, tranks);
    emxFree_real_T(&b_Vs);
  } else {
    c_emxInit_real_T(&b_Vs, 3);

    /* 'constrained_smooth_surf_cleanmesh:10' else */
    /* 'constrained_smooth_surf_cleanmesh:11' [nrms_surf, Vs] = eigenanalysis_surf( Vs, bs_m, isridge); */
    k = b_Vs->size[0] * b_Vs->size[1] * b_Vs->size[2];
    b_Vs->size[0] = 3;
    b_Vs->size[1] = 3;
    b_Vs->size[2] = Vs->size[2];
    emxEnsureCapacity((emxArray__common *)b_Vs, k, (int32_T)sizeof(real_T));
    ix = Vs->size[0] * Vs->size[1] * Vs->size[2] - 1;
    for (k = 0; k <= ix; k++) {
      b_Vs->data[k] = Vs->data[k];
    }

    b_eigenanalysis_surf(b_Vs, bs_m, isridge, nrms_surf, Vs);

    /* 'constrained_smooth_surf_cleanmesh:12' tranks = zeros(size(xs,1),1,'int8'); */
    k = tranks->size[0];
    tranks->size[0] = xs->size[0];
    emxEnsureCapacity((emxArray__common *)tranks, k, (int32_T)sizeof(int8_T));
    emxFree_real_T(&b_Vs);
    ix = xs->size[0] - 1;
    for (k = 0; k <= ix; k++) {
      tranks->data[k] = 0;
    }
  }

  emxFree_real_T(&bs_m);
  emxInit_boolean_T(&isfree, 1);

  /* 'constrained_smooth_surf_cleanmesh:15' nv = int32(size(xs,1)); */
  nv = xs->size[0];

  /*  Determine boundary vertices */
  /*  isfree = ~determine_border_vertices_surf( nv, tris, determine_opposite_halfedge_tri(nv, tris)); */
  /* 'constrained_smooth_surf_cleanmesh:20' isfree = false(nv,1); */
  k = isfree->size[0];
  isfree->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)isfree, k, (int32_T)sizeof(boolean_T));
  ix = nv - 1;
  for (k = 0; k <= ix; k++) {
    isfree->data[k] = FALSE;
  }

  /* 'constrained_smooth_surf_cleanmesh:21' for jj=1:int32(size(tris,1)) */
  for (jj = 1; jj <= tris->size[0]; jj++) {
    /* 'constrained_smooth_surf_cleanmesh:21' isfree(tris(jj,1:3))=true; */
    for (k = 0; k < 3; k++) {
      isfree->data[tris->data[(jj + tris->size[0] * k) - 1] - 1] = TRUE;
    }
  }

  /* 'constrained_smooth_surf_cleanmesh:23' us_smooth = zeros(nv,3); */
  k = us_smooth->size[0] * us_smooth->size[1];
  us_smooth->size[0] = nv;
  us_smooth->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)us_smooth, k, (int32_T)sizeof(real_T));
  ix = nv * 3 - 1;
  for (k = 0; k <= ix; k++) {
    us_smooth->data[k] = 0.0;
  }

  /*  Project grad and hess onto tangent space */
  /* 'constrained_smooth_surf_cleanmesh:26' for jj=1:nv_clean */
  for (jj = 0; jj + 1 <= nv_clean; jj++) {
    /* 'constrained_smooth_surf_cleanmesh:27' if ~isfree(jj) */
    if (!isfree->data[jj]) {
    } else {
      /* 'constrained_smooth_surf_cleanmesh:29' if numel(isridge)>=nv && isridge(jj) || check_trank && tranks(jj)==1 */
      if (((isridge->size[0] >= nv) && isridge->data[jj]) || (check_trank &&
           (tranks->data[jj] == 1))) {
        /* 'constrained_smooth_surf_cleanmesh:30' t = Vs(:,3,jj); */
        /*  Smooth only along ridge directions */
        /* 'constrained_smooth_surf_cleanmesh:33' h = t'*Hs_smooth(:,:,jj)*t; */
        for (k = 0; k < 3; k++) {
          y[k] = 0.0;
          for (ix = 0; ix < 3; ix++) {
            h = y[k] + Vs->data[(ix + (Vs->size[0] << 1)) + Vs->size[0] *
              Vs->size[1] * jj] * Hs_smooth->data[(ix + Hs_smooth->size[0] * k)
              + Hs_smooth->size[0] * Hs_smooth->size[1] * jj];
            y[k] = h;
          }
        }

        h = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          h += y[ix] * Vs->data[(iy + (Vs->size[0] << 1)) + Vs->size[0] *
            Vs->size[1] * jj];
          ix++;
          iy++;
        }

        /* 'constrained_smooth_surf_cleanmesh:34' g = grads_smooth(:,jj)'*t; */
        for (k = 0; k < 3; k++) {
          y[k] = grads_smooth->data[k + grads_smooth->size[0] * jj];
        }

        g = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          g += y[ix] * Vs->data[(iy + (Vs->size[0] << 1)) + Vs->size[0] *
            Vs->size[1] * jj];
          ix++;
          iy++;
        }

        /*  Project Hessian and gradient onto tangent space */
        /* 'constrained_smooth_surf_cleanmesh:37' us_smooth(jj,:) = -g/h*t'; */
        h = -g / h;
        for (k = 0; k < 3; k++) {
          us_smooth->data[jj + us_smooth->size[0] * k] = h * Vs->data[(k +
            (Vs->size[0] << 1)) + Vs->size[0] * Vs->size[1] * jj];
        }
      } else if ((!check_trank) || (tranks->data[jj] == 2)) {
        /* 'constrained_smooth_surf_cleanmesh:38' elseif ~check_trank || tranks(jj)==2 */
        /* 'constrained_smooth_surf_cleanmesh:39' nrm = nrms_surf(jj,:)'; */
        for (k = 0; k < 3; k++) {
          nrm[k] = nrms_surf->data[jj + nrms_surf->size[0] * k];
        }

        /* 'constrained_smooth_surf_cleanmesh:39' nrm = nrm / sqrt(nrm'*nrm+1.e-100); */
        h = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          h += nrm[ix] * nrm[iy];
          ix++;
          iy++;
        }

        g = sqrt(h + 1.0E-100);

        /* 'constrained_smooth_surf_cleanmesh:40' T = obtain_tangents_surf( nrm); */
        /* OBTAIN_TANGENTS_SURF Obtain orthonormal tangent vectors from given unit  */
        /* normal. */
        /*  T = OBTAIN_TANGENTS_SURF(NRM) Obtains orthonormal tangent vectors in 3x1  */
        /*  T from given 3x1 unit normal NRM.  */
        /* 'obtain_tangents_surf:7' coder.inline('always') */
        /* 'obtain_tangents_surf:9' assert( 1.-nrm'*nrm < 1.e-10); */
        /* 'obtain_tangents_surf:10' absnrm = abs(nrm); */
        for (k = 0; k < 3; k++) {
          h = nrm[k] / g;
          absnrm[k] = fabs(h);
          nrm[k] = h;
        }

        /* 'obtain_tangents_surf:12' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
        if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
          /* 'obtain_tangents_surf:13' t1 = nrm(2) * nrm; */
          for (ix = 0; ix < 3; ix++) {
            absnrm[ix] = nrm[1] * nrm[ix];
          }

          /* 'obtain_tangents_surf:13' t1(2) = t1(2) -1; */
          absnrm[1]--;
        } else {
          /* 'obtain_tangents_surf:14' else */
          /* 'obtain_tangents_surf:15' t1 = nrm(1) * nrm; */
          for (ix = 0; ix < 3; ix++) {
            absnrm[ix] = nrm[0] * nrm[ix];
          }

          /* 'obtain_tangents_surf:15' t1(1) = t1(1) -1; */
          absnrm[0]--;
        }

        /* 'obtain_tangents_surf:17' t1 = t1 / sqrt(t1'*t1); */
        h = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          h += absnrm[ix] * absnrm[iy];
          ix++;
          iy++;
        }

        g = sqrt(h);

        /* 'obtain_tangents_surf:19' T=[t1,cross_col( t1, nrm)]; */
        /* CROSS_COL Efficient routine for computing cross product of two  */
        /* 3-dimensional column vectors. */
        /*  CROSS_COL(A,B) Efficiently computes the cross product between */
        /*  3-dimensional column vector A, and 3-dimensional column vector B. */
        /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
        for (k = 0; k < 3; k++) {
          h = absnrm[k] / g;
          T[k] = h;
          absnrm[k] = h;
        }

        b_absnrm[0] = absnrm[1] * nrm[2] - absnrm[2] * nrm[1];
        b_absnrm[1] = absnrm[2] * nrm[0] - absnrm[0] * nrm[2];
        b_absnrm[2] = absnrm[0] * nrm[1] - absnrm[1] * nrm[0];
        for (k = 0; k < 3; k++) {
          T[3 + k] = b_absnrm[k];
        }

        /* 'constrained_smooth_surf_cleanmesh:42' H2 = T'*Hs_smooth(:,:,jj)*T; */
        for (k = 0; k < 2; k++) {
          for (ix = 0; ix < 3; ix++) {
            b_T[k + (ix << 1)] = 0.0;
            for (iy = 0; iy < 3; iy++) {
              b_T[k + (ix << 1)] += T[iy + 3 * k] * Hs_smooth->data[(iy +
                Hs_smooth->size[0] * ix) + Hs_smooth->size[0] * Hs_smooth->size
                [1] * jj];
            }
          }
        }

        for (k = 0; k < 2; k++) {
          for (ix = 0; ix < 2; ix++) {
            H2[k + (ix << 1)] = 0.0;
            for (iy = 0; iy < 3; iy++) {
              H2[k + (ix << 1)] += b_T[k + (iy << 1)] * T[iy + 3 * ix];
            }
          }
        }

        /* 'constrained_smooth_surf_cleanmesh:43' [R2, err] = chol2(H2); */
        /*  CHOL2 Perform cholesky factorization of a 2x2 SPD matrix. */
        /*    R=CHOL2(A). */
        /*    A is a 2-by-2 SPD matrix. Only its upper triangular part is accessed. */
        /*  */
        /*  See also CHOL3. */
        /* 'chol2:8' R = zeros(2,2); */
        for (k = 0; k < 4; k++) {
          R2[k] = 0.0;
        }

        /* 'chol2:10' if A(1,1)<=0 */
        if (H2[0] <= 0.0) {
          /* 'chol2:10' err = -1; */
          ix = -1;
        } else {
          /* 'chol2:11' R(1,1) = sqrt(A(1,1)); */
          R2[0] = sqrt(H2[0]);

          /* 'chol2:12' R(1,2) = A(1,2) / R(1,1); */
          R2[2] = H2[2] / R2[0];

          /* 'chol2:13' R(2,2) = A(2,2) - R(1,2)*R(1,2); */
          R2[3] = H2[3] - R2[2] * R2[2];

          /* 'chol2:15' if R(2,2)<=0 */
          if (R2[3] <= 0.0) {
            /* 'chol2:15' err = -1; */
            ix = -1;
          } else {
            /* 'chol2:16' R(2,2) = sqrt(R(2,2)); */
            R2[3] = sqrt(R2[3]);

            /* 'chol2:18' err = 0; */
            ix = 0;
          }
        }

        /* 'constrained_smooth_surf_cleanmesh:45' if ~err */
        if (!(ix != 0)) {
          /* 'constrained_smooth_surf_cleanmesh:46' g2 = T'*(-grads_smooth(:,jj)); */
          for (k = 0; k < 3; k++) {
            absnrm[k] = -grads_smooth->data[k + grads_smooth->size[0] * jj];
          }

          for (k = 0; k < 2; k++) {
            g2[k] = 0.0;
            for (ix = 0; ix < 3; ix++) {
              g2[k] += T[ix + 3 * k] * absnrm[ix];
            }
          }

          /* 'constrained_smooth_surf_cleanmesh:47' us_smooth(jj,:) = T*backsolve( R2,forwardsolve_trans(R2, g2)); */
          /*  Perform forward substitution R'\bs. */
          /*      bs = forwardsolve_trans(R, bs) */
          /*      bs = forwardsolve_trans(R, bs, cend) */
          /*      bs = forwardsolve_trans(R, bs, cend, ws) */
          /*   Compute bs = (triu(R(1:cend,1:cend))'\bs) ./ ws; */
          /*   The right-hand side vector bs can have multiple columns. */
          /*   By default, cend is size(R,2), and ws is ones. */
          /* 'forwardsolve_trans:10' if nargin<3 */
          /* 'forwardsolve_trans:10' cend = int32(size(R,2)); */
          /* 'forwardsolve_trans:12' for kk=1:int32(size(bs,2)) */
          /*  Skip zeros in bs */
          /* 'forwardsolve_trans:14' cstart = cend+1; */
          ix = 3;

          /* 'forwardsolve_trans:15' for ii=1:cend */
          iy = 1;
          exitg1 = 0U;
          while ((exitg1 == 0U) && (iy < 3)) {
            /* 'forwardsolve_trans:16' if (bs(ii)~=0) */
            if (g2[iy - 1] != 0.0) {
              /* 'forwardsolve_trans:17' cstart = ii; */
              ix = iy;
              exitg1 = 1U;
            } else {
              iy++;
            }
          }

          /* 'forwardsolve_trans:22' for jj=cstart:1:cend */
          for (k = ix - 1; k + 1 < 3; k++) {
            /* 'forwardsolve_trans:23' for ii=cstart:jj-1 */
            iy = ix;
            while (iy <= k) {
              /* 'forwardsolve_trans:24' bs(jj,kk) = bs(jj,kk) - R(ii,jj) * bs(ii,kk); */
              g2[k] -= R2[k << 1] * g2[0];
              iy = 2;
            }

            /* 'forwardsolve_trans:27' assert( R(jj,jj)~=0); */
            /* 'forwardsolve_trans:28' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
            g2[k] /= R2[k + (k << 1)];
          }

          /* 'forwardsolve_trans:32' if nargin>3 */
          /*  Perform backward substitution. */
          /*      bs = backsolve(R, bs) */
          /*      bs = backsolve(R, bs, cend) */
          /*      bs = backsolve(R, bs, cend, ws) */
          /*   Compute bs = (triu(R(1:cend,1:cend))\bs) ./ ws; */
          /*   The right-hand side vector bs can have multiple columns. */
          /*   By default, cend is size(R,2), and ws is ones. */
          /* 'backsolve:10' if nargin<3 */
          /* 'backsolve:10' cend = int32(size(R,2)); */
          /* 'backsolve:12' for kk=1:int32(size(bs,2)) */
          /* 'backsolve:13' for jj=cend:-1:1 */
          for (k = 1; k > -1; k += -1) {
            /* 'backsolve:14' for ii=jj+1:cend */
            iy = k + 2;
            while (iy < 3) {
              /* 'backsolve:15' bs(jj,kk) = bs(jj,kk) - R(jj,ii) * bs(ii,kk); */
              g2[k] -= R2[2 + k] * g2[1];
              iy = 3;
            }

            /* 'backsolve:18' assert( R(jj,jj)~=0); */
            /* 'backsolve:19' bs(jj,kk) = bs(jj,kk) / R(jj,jj); */
            g2[k] /= R2[k + (k << 1)];
          }

          /* 'backsolve:23' if nargin>3 */
          for (k = 0; k < 3; k++) {
            nrm[k] = 0.0;
            for (ix = 0; ix < 2; ix++) {
              nrm[k] += T[k + 3 * ix] * g2[ix];
            }
          }

          for (k = 0; k < 3; k++) {
            us_smooth->data[jj + us_smooth->size[0] * k] = nrm[k];
          }
        }
      } else {
        /* 'constrained_smooth_surf_cleanmesh:49' else */
        /*  Otherwise, tranks(jj)=0, and us_smooth should be zero. */
      }

      /* 'constrained_smooth_surf_cleanmesh:53' if any(ns_constrained(:,jj)~=0) */
      for (k = 0; k < 3; k++) {
        b_ns_constrained[k] = (ns_constrained->data[k + ns_constrained->size[0] *
          jj] != 0.0);
      }

      if (any(b_ns_constrained)) {
        /* 'constrained_smooth_surf_cleanmesh:54' nrm = ns_constrained(:,jj); */
        /* 'constrained_smooth_surf_cleanmesh:55' us_smooth(jj,:) = us_smooth(jj,:) - (us_smooth(jj,:)/(nrm'*nrm)*nrm) * nrm'; */
        h = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          h += ns_constrained->data[ix + ns_constrained->size[0] * jj] *
            ns_constrained->data[iy + ns_constrained->size[0] * jj];
          ix++;
          iy++;
        }

        for (k = 0; k < 3; k++) {
          y[k] = us_smooth->data[jj + us_smooth->size[0] * k] / h;
        }

        h = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          h += y[ix] * ns_constrained->data[iy + ns_constrained->size[0] * jj];
          ix++;
          iy++;
        }

        for (k = 0; k < 3; k++) {
          us_smooth->data[jj + us_smooth->size[0] * k] -= h *
            ns_constrained->data[k + ns_constrained->size[0] * jj];
        }
      }
    }
  }

  emxFree_real_T(&ns_constrained);
  emxFree_real_T(&Vs);
  emxFree_boolean_T(&isfree);
  emxFree_int8_T(&tranks);
  emxFree_real_T(&nrms_surf);
}

/*
 *
 */
static boolean_T c_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv2[8] = { 'w', 'a', 'l', 'f', '_', 'n', 'r', 'm' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(7 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv2[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
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
 * function [bs, degree] = eval_vander_bivar( us, bs, degree, ws, interp0, guardosc)
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
  /*  [BS,DEGREE] = EVAL_VANDER_BIVAR(US,BS,DEGREE,WS, INTERP, GUARDOSC)  */
  /*  Evaluates generalized Vandermonde matrix V, and solve V\BS. */
  /*  It supports up to degree 6. */
  /*   */
  /*  If interp0 is true, then the fitting is forced to pass through origin. */
  /*  */
  /*  See also EVAL_VANDER_UNIVAR */
  /* 'eval_vander_bivar:10' degree = int32(degree); */
  /* 'eval_vander_bivar:11' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:14' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:15' if nargin<5 */
  /* 'eval_vander_bivar:16' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:19' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  ncols = (*degree + 2) * (*degree + 1) / 2;

  /* 'eval_vander_bivar:20' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (*degree > 1)) {
    /* 'eval_vander_bivar:21' degree=degree-1; */
    (*degree)--;

    /* 'eval_vander_bivar:22' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
    ncols = (*degree + 2) * (*degree + 1) / 2;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  /* 'eval_vander_bivar:26' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, *degree, V);

  /* 'eval_vander_bivar:27' if interp0 */
  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar:30' if nargin>3 && ~isempty(ws) */
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:31' if degree>2 */
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      /* 'eval_vander_bivar:32' ws1 = us(:,1).*us(:,1)+us(:,2).*us(:,2); */
      jj = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] = us->data[jj] * us->data[jj] + us->data[jj + us->size[0]]
          * us->data[jj + us->size[0]];
      }

      /* 'eval_vander_bivar:33' ws1 = ws1 + sum(ws1)/double(npnts)*1.e-2; */
      A = sum(ws1);
      A = A / (real_T)npnts * 0.01;
      jj = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (jj = 0; jj <= loop_ub; jj++) {
        ws1->data[jj] += A;
      }

      /* 'eval_vander_bivar:34' if degree<4 */
      if (*degree < 4) {
        /* 'eval_vander_bivar:35' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:36' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:37' ws1(ii) = ws(ii) / sqrt(ws1(ii)); */
            ws1->data[ii] = ws->data[ii] / sqrt(ws1->data[ii]);
          } else {
            /* 'eval_vander_bivar:38' else */
            /* 'eval_vander_bivar:39' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      } else {
        /* 'eval_vander_bivar:42' else */
        /* 'eval_vander_bivar:43' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:44' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:45' ws1(ii) = ws(ii) / ws1(ii); */
            ws1->data[ii] = ws->data[ii] / ws1->data[ii];
          } else {
            /* 'eval_vander_bivar:46' else */
            /* 'eval_vander_bivar:47' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      }

      /* 'eval_vander_bivar:51' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:52' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:52' V(ii,jj) = V(ii,jj) * ws1(ii); */
          V->data[ii + V->size[0] * jj] *= ws1->data[ii];
        }

        /* 'eval_vander_bivar:53' for jj=1:size(bs,2) */
        for (jj = 0; jj < 2; jj++) {
          /* 'eval_vander_bivar:53' bs(ii,jj) = bs(ii,jj) * ws1(ii); */
          bs->data[ii + bs->size[0] * jj] *= ws1->data[ii];
        }
      }
    } else {
      /* 'eval_vander_bivar:55' else */
      /* 'eval_vander_bivar:56' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:57' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:57' V(ii,jj) = V(ii,jj) * ws(ii); */
          V->data[ii + V->size[0] * jj] *= ws->data[ii];
        }

        /* 'eval_vander_bivar:58' for jj=1:int32(size(bs,2)) */
        for (jj = 0; jj < 2; jj++) {
          /* 'eval_vander_bivar:58' bs(ii,jj) = bs(ii,jj) * ws(ii); */
          bs->data[ii + bs->size[0] * jj] *= ws->data[ii];
        }
      }
    }
  }

  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:65' ts = coder.nullcopy(zeros(ncols,1)); */
  jj = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, jj, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:66' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:69' D = coder.nullcopy(zeros(ncols,1)); */
  jj = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, jj, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:70' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  ii = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:73' ncols_sub = ncols; */
  /* 'eval_vander_bivar:74' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (ii < ncols) {
      /* 'eval_vander_bivar:75' degree = degree-1; */
      (*degree)--;

      /* 'eval_vander_bivar:77' if degree==0 */
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar:79' bs(:) = 0; */
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
        /* 'eval_vander_bivar:81' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U);
      }
    } else {
      /* % Compute Q'bs */
      /* 'eval_vander_bivar:85' bs = compute_qtb( V, bs, ncols_sub); */
      b_compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar:88' for i=1:ncols_sub */
      for (ii = 0; ii + 1 <= ncols; ii++) {
        /* 'eval_vander_bivar:88' V(i,i) = D(i); */
        V->data[ii + V->size[0] * ii] = D->data[ii];
      }

      /* 'eval_vander_bivar:89' if guardosc */
      /* 'eval_vander_bivar:91' else */
      /* 'eval_vander_bivar:92' bs = backsolve(V, bs, ncols_sub, ts); */
      b_backsolve(V, bs, ncols, ws1);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
}

/*
 * function [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point...
 *     ( v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
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
 * function us_smooth = weighted_Laplacian_tri_cleanmesh(nv_clean, xs, tris, isridge, ridgeedge, flabel, check_trank)
 */
static void c_weighted_Laplacian_tri_cleanm(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_boolean_T
  *isridge, const emxArray_boolean_T *ridgeedge, const emxArray_int32_T *flabel,
  boolean_T check_trank, emxArray_real_T *us_smooth)
{
  emxArray_real_T *fcenters;
  int32_T nv;
  int32_T ntri;
  int32_T i3;
  int32_T ix;
  uint32_T jj;
  int32_T b_jj;
  int32_T k;
  real_T b_xs[9];
  real_T disp[3];
  emxArray_real_T *ws_smooth;
  int32_T kk;
  real_T w;
  real_T nrm[3];
  static const int8_T iv14[3] = { 2, 3, 1 };

  emxArray_real_T *b_us_smooth;
  emxArray_real_T *b_ws_smooth;
  emxArray_real_T *r6;
  emxArray_real_T *c_us_smooth;
  emxArray_real_T *c_ws_smooth;
  emxArray_real_T *d_us_smooth;
  emxArray_real_T *d_ws_smooth;
  emxArray_real_T *Vs;
  emxArray_real_T *bs_m;
  emxArray_real_T *ns_constrained;
  emxArray_int8_T *tranks;
  emxArray_real_T *b_Vs;
  boolean_T b_ns_constrained[3];
  real_T y;
  emxInit_real_T(&fcenters, 2);

  /* % Computed distance-weighted averaging of face normal */
  /*  Loop through faces to compute a smoothing term */
  /*  Compute all face centers */
  /* 'weighted_Laplacian_tri_cleanmesh:9' coder.inline('never') */
  /* 'weighted_Laplacian_tri_cleanmesh:10' if nargin<4 */
  /* 'weighted_Laplacian_tri_cleanmesh:11' if nargin<5 */
  /* 'weighted_Laplacian_tri_cleanmesh:12' if nargin<6 */
  /* 'weighted_Laplacian_tri_cleanmesh:13' if nargin<7 */
  /* 'weighted_Laplacian_tri_cleanmesh:15' nv   = size(xs,1); */
  nv = xs->size[0];

  /* 'weighted_Laplacian_tri_cleanmesh:16' ntri = size(tris,1); */
  ntri = tris->size[0];

  /* 'weighted_Laplacian_tri_cleanmesh:18' fcenters = zeros(ntri,3); */
  i3 = fcenters->size[0] * fcenters->size[1];
  fcenters->size[0] = ntri;
  fcenters->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)fcenters, i3, (int32_T)sizeof(real_T));
  ix = ntri * 3 - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    fcenters->data[i3] = 0.0;
  }

  /* 'weighted_Laplacian_tri_cleanmesh:19' for jj=1:ntri */
  for (jj = 1U; jj <= (uint32_T)ntri; jj++) {
    /* 'weighted_Laplacian_tri_cleanmesh:20' fcenters(jj,:) = sum(xs(tris(jj,:),:),1)/3; */
    b_jj = (int32_T)jj;
    for (i3 = 0; i3 < 3; i3++) {
      for (k = 0; k < 3; k++) {
        b_xs[k + 3 * i3] = xs->data[(tris->data[(b_jj + tris->size[0] * k) - 1]
          + xs->size[0] * i3) - 1];
      }
    }

    b_sum(b_xs, disp);
    b_jj = (int32_T)jj;
    for (i3 = 0; i3 < 3; i3++) {
      fcenters->data[(b_jj + fcenters->size[0] * i3) - 1] = disp[i3] / 3.0;
    }
  }

  /* 'weighted_Laplacian_tri_cleanmesh:23' us_smooth = zeros(nv,3); */
  i3 = us_smooth->size[0] * us_smooth->size[1];
  us_smooth->size[0] = nv;
  us_smooth->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)us_smooth, i3, (int32_T)sizeof(real_T));
  ix = nv * 3 - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    us_smooth->data[i3] = 0.0;
  }

  b_emxInit_real_T(&ws_smooth, 1);

  /* 'weighted_Laplacian_tri_cleanmesh:24' ws_smooth = repmat(1.e-100,nv,1); */
  repmat((real_T)nv, ws_smooth);

  /* 'weighted_Laplacian_tri_cleanmesh:26' next = [2 3 1]; */
  /* 'weighted_Laplacian_tri_cleanmesh:27' for jj=1:ntri */
  for (jj = 1U; jj <= (uint32_T)ntri; jj++) {
    /* 'weighted_Laplacian_tri_cleanmesh:28' center = fcenters(jj,:); */
    /* 'weighted_Laplacian_tri_cleanmesh:29' for kk=1:3 */
    for (kk = 0; kk < 3; kk++) {
      /* 'weighted_Laplacian_tri_cleanmesh:30' v = tris(jj,kk); */
      /* 'weighted_Laplacian_tri_cleanmesh:31' if v>nv_clean */
      if (tris->data[((int32_T)jj + tris->size[0] * kk) - 1] > nv_clean) {
      } else {
        /* 'weighted_Laplacian_tri_cleanmesh:34' if isempty(isridge) || ~isridge(v) */
        if ((isridge->size[0] == 0) || (!isridge->data[tris->data[((int32_T)jj +
              tris->size[0] * kk) - 1] - 1])) {
          /* 'weighted_Laplacian_tri_cleanmesh:35' disp = center-xs(v,:); */
          b_jj = (int32_T)jj;
          nv = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
          for (i3 = 0; i3 < 3; i3++) {
            disp[i3] = fcenters->data[(b_jj + fcenters->size[0] * i3) - 1] -
              xs->data[(nv + xs->size[0] * i3) - 1];
          }

          /* 'weighted_Laplacian_tri_cleanmesh:35' w=sqrt(disp*disp'); */
          w = 0.0;
          ix = 0;
          nv = 0;
          for (k = 0; k < 3; k++) {
            w += disp[ix] * disp[nv];
            ix++;
            nv++;
          }

          w = sqrt(w);

          /* 'weighted_Laplacian_tri_cleanmesh:37' us_smooth(v,1:3) = us_smooth(v,1:3) + disp*w; */
          nv = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
          ix = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
          for (i3 = 0; i3 < 3; i3++) {
            nrm[i3] = us_smooth->data[(ix + us_smooth->size[0] * i3) - 1] +
              disp[i3] * w;
          }

          for (i3 = 0; i3 < 3; i3++) {
            us_smooth->data[(nv + us_smooth->size[0] * i3) - 1] = nrm[i3];
          }

          /* 'weighted_Laplacian_tri_cleanmesh:38' ws_smooth(v) = ws_smooth(v) + w; */
          ws_smooth->data[tris->data[((int32_T)jj + tris->size[0] * kk) - 1] - 1]
            += w;
        } else {
          if ((!(ridgeedge->size[0] == 0)) && ridgeedge->data[((int32_T)jj +
               ridgeedge->size[0] * kk) - 1]) {
            /* 'weighted_Laplacian_tri_cleanmesh:39' elseif ~isempty(ridgeedge) && ridgeedge(jj,kk) */
            /* 'weighted_Laplacian_tri_cleanmesh:40' disp = xs(tris(jj,next(kk)),:)-xs(v,:); */
            nv = tris->data[((int32_T)jj + tris->size[0] * (iv14[kk] - 1)) - 1];
            ix = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
            for (i3 = 0; i3 < 3; i3++) {
              disp[i3] = xs->data[(nv + xs->size[0] * i3) - 1] - xs->data[(ix +
                xs->size[0] * i3) - 1];
            }

            /* 'weighted_Laplacian_tri_cleanmesh:40' w=1; */
            /* 'weighted_Laplacian_tri_cleanmesh:41' us_smooth(v,1:3) = us_smooth(v,1:3) + disp(1:3).*w; */
            nv = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
            ix = tris->data[((int32_T)jj + tris->size[0] * kk) - 1];
            for (i3 = 0; i3 < 3; i3++) {
              nrm[i3] = us_smooth->data[(ix + us_smooth->size[0] * i3) - 1] +
                disp[i3];
            }

            for (i3 = 0; i3 < 3; i3++) {
              us_smooth->data[(nv + us_smooth->size[0] * i3) - 1] = nrm[i3];
            }

            /* 'weighted_Laplacian_tri_cleanmesh:42' ws_smooth(v) = ws_smooth(v) + 2*w; */
            ws_smooth->data[tris->data[((int32_T)jj + tris->size[0] * kk) - 1] -
              1] += 2.0;
          }
        }
      }
    }
  }

  /*  Compute smoothing term and project onto constrained surface */
  /* 'weighted_Laplacian_tri_cleanmesh:50' us_smooth(1:nv_clean,1) = us_smooth(1:nv_clean,1) ./ ws_smooth(1:nv_clean); */
  if (1 > nv_clean) {
    i3 = 0;
  } else {
    i3 = nv_clean;
  }

  if (1 > nv_clean) {
    k = 0;
  } else {
    k = nv_clean;
  }

  b_emxInit_real_T(&b_us_smooth, 1);
  ix = b_us_smooth->size[0];
  b_us_smooth->size[0] = i3;
  emxEnsureCapacity((emxArray__common *)b_us_smooth, ix, (int32_T)sizeof(real_T));
  ix = i3 - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    b_us_smooth->data[i3] = us_smooth->data[i3];
  }

  b_emxInit_real_T(&b_ws_smooth, 1);
  i3 = b_ws_smooth->size[0];
  b_ws_smooth->size[0] = k;
  emxEnsureCapacity((emxArray__common *)b_ws_smooth, i3, (int32_T)sizeof(real_T));
  ix = k - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    b_ws_smooth->data[i3] = ws_smooth->data[i3];
  }

  b_emxInit_real_T(&r6, 1);
  rdivide(b_us_smooth, b_ws_smooth, r6);
  emxFree_real_T(&b_ws_smooth);
  emxFree_real_T(&b_us_smooth);
  ix = r6->size[0] - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    us_smooth->data[i3] = r6->data[i3];
  }

  /* 'weighted_Laplacian_tri_cleanmesh:51' us_smooth(1:nv_clean,2) = us_smooth(1:nv_clean,2) ./ ws_smooth(1:nv_clean); */
  if (1 > nv_clean) {
    i3 = 0;
  } else {
    i3 = nv_clean;
  }

  if (1 > nv_clean) {
    k = 0;
  } else {
    k = nv_clean;
  }

  b_emxInit_real_T(&c_us_smooth, 1);
  ix = c_us_smooth->size[0];
  c_us_smooth->size[0] = i3;
  emxEnsureCapacity((emxArray__common *)c_us_smooth, ix, (int32_T)sizeof(real_T));
  ix = i3 - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    c_us_smooth->data[i3] = us_smooth->data[i3 + us_smooth->size[0]];
  }

  b_emxInit_real_T(&c_ws_smooth, 1);
  i3 = c_ws_smooth->size[0];
  c_ws_smooth->size[0] = k;
  emxEnsureCapacity((emxArray__common *)c_ws_smooth, i3, (int32_T)sizeof(real_T));
  ix = k - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    c_ws_smooth->data[i3] = ws_smooth->data[i3];
  }

  rdivide(c_us_smooth, c_ws_smooth, r6);
  emxFree_real_T(&c_ws_smooth);
  emxFree_real_T(&c_us_smooth);
  ix = r6->size[0] - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    us_smooth->data[i3 + us_smooth->size[0]] = r6->data[i3];
  }

  /* 'weighted_Laplacian_tri_cleanmesh:52' us_smooth(1:nv_clean,3) = us_smooth(1:nv_clean,3) ./ ws_smooth(1:nv_clean); */
  if (1 > nv_clean) {
    i3 = 0;
  } else {
    i3 = nv_clean;
  }

  if (1 > nv_clean) {
    k = 0;
  } else {
    k = nv_clean;
  }

  b_emxInit_real_T(&d_us_smooth, 1);
  ix = d_us_smooth->size[0];
  d_us_smooth->size[0] = i3;
  emxEnsureCapacity((emxArray__common *)d_us_smooth, ix, (int32_T)sizeof(real_T));
  ix = i3 - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    d_us_smooth->data[i3] = us_smooth->data[i3 + (us_smooth->size[0] << 1)];
  }

  b_emxInit_real_T(&d_ws_smooth, 1);
  i3 = d_ws_smooth->size[0];
  d_ws_smooth->size[0] = k;
  emxEnsureCapacity((emxArray__common *)d_ws_smooth, i3, (int32_T)sizeof(real_T));
  ix = k - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    d_ws_smooth->data[i3] = ws_smooth->data[i3];
  }

  emxFree_real_T(&ws_smooth);
  rdivide(d_us_smooth, d_ws_smooth, r6);
  emxFree_real_T(&d_ws_smooth);
  emxFree_real_T(&d_us_smooth);
  ix = r6->size[0] - 1;
  for (i3 = 0; i3 <= ix; i3++) {
    us_smooth->data[i3 + (us_smooth->size[0] << 1)] = r6->data[i3];
  }

  emxFree_real_T(&r6);
  c_emxInit_real_T(&Vs, 3);
  emxInit_real_T(&bs_m, 2);
  emxInit_real_T(&ns_constrained, 2);

  /*  Project displacement onto tangent space and smooth along normal direction */
  /* 'weighted_Laplacian_tri_cleanmesh:55' [Vs, bs_m, ns_constrained] = compute_medial_quadric_tri( xs, tris, flabel); */
  compute_medial_quadric_tri(xs, tris, flabel, Vs, bs_m, ns_constrained);

  /* 'weighted_Laplacian_tri_cleanmesh:56' if check_trank */
  emxInit_int8_T(&tranks, 1);
  if (check_trank) {
    c_emxInit_real_T(&b_Vs, 3);

    /* 'weighted_Laplacian_tri_cleanmesh:57' [nrm_surf, Vs, tranks] = eigenanalysis_surf( Vs, bs_m, isridge); */
    i3 = b_Vs->size[0] * b_Vs->size[1] * b_Vs->size[2];
    b_Vs->size[0] = 3;
    b_Vs->size[1] = 3;
    b_Vs->size[2] = Vs->size[2];
    emxEnsureCapacity((emxArray__common *)b_Vs, i3, (int32_T)sizeof(real_T));
    ix = Vs->size[0] * Vs->size[1] * Vs->size[2] - 1;
    for (i3 = 0; i3 <= ix; i3++) {
      b_Vs->data[i3] = Vs->data[i3];
    }

    eigenanalysis_surf(b_Vs, bs_m, isridge, fcenters, Vs, tranks);
    emxFree_real_T(&b_Vs);
  } else {
    c_emxInit_real_T(&b_Vs, 3);

    /* 'weighted_Laplacian_tri_cleanmesh:58' else */
    /* 'weighted_Laplacian_tri_cleanmesh:59' [nrm_surf, Vs] = eigenanalysis_surf( Vs, bs_m, isridge); */
    i3 = b_Vs->size[0] * b_Vs->size[1] * b_Vs->size[2];
    b_Vs->size[0] = 3;
    b_Vs->size[1] = 3;
    b_Vs->size[2] = Vs->size[2];
    emxEnsureCapacity((emxArray__common *)b_Vs, i3, (int32_T)sizeof(real_T));
    ix = Vs->size[0] * Vs->size[1] * Vs->size[2] - 1;
    for (i3 = 0; i3 <= ix; i3++) {
      b_Vs->data[i3] = Vs->data[i3];
    }

    b_eigenanalysis_surf(b_Vs, bs_m, isridge, fcenters, Vs);

    /* 'weighted_Laplacian_tri_cleanmesh:60' tranks = zeros(size(xs,1),1,'int8'); */
    i3 = tranks->size[0];
    tranks->size[0] = xs->size[0];
    emxEnsureCapacity((emxArray__common *)tranks, i3, (int32_T)sizeof(int8_T));
    emxFree_real_T(&b_Vs);
    ix = xs->size[0] - 1;
    for (i3 = 0; i3 <= ix; i3++) {
      tranks->data[i3] = 0;
    }
  }

  emxFree_real_T(&bs_m);

  /* 'weighted_Laplacian_tri_cleanmesh:63' for jj=1:nv_clean */
  for (b_jj = 0; b_jj + 1 <= nv_clean; b_jj++) {
    /* 'weighted_Laplacian_tri_cleanmesh:64' if ~isempty(isridge) && isridge(jj) || check_trank && tranks(jj)==1 */
    if (((!(isridge->size[0] == 0)) && isridge->data[b_jj]) || (check_trank &&
         (tranks->data[b_jj] == 1))) {
      /* 'weighted_Laplacian_tri_cleanmesh:65' t = Vs(:,3,jj); */
      /* 'weighted_Laplacian_tri_cleanmesh:66' us_smooth(jj,:) = (us_smooth(jj,:)*t)*t'; */
      w = 0.0;
      ix = 0;
      nv = 0;
      for (k = 0; k < 3; k++) {
        w += us_smooth->data[b_jj + us_smooth->size[0] * ix] * Vs->data[(nv +
          (Vs->size[0] << 1)) + Vs->size[0] * Vs->size[1] * b_jj];
        ix++;
        nv++;
      }

      for (i3 = 0; i3 < 3; i3++) {
        us_smooth->data[b_jj + us_smooth->size[0] * i3] = w * Vs->data[(i3 +
          (Vs->size[0] << 1)) + Vs->size[0] * Vs->size[1] * b_jj];
      }
    } else {
      if (check_trank && (tranks->data[b_jj] == 0)) {
        /* 'weighted_Laplacian_tri_cleanmesh:67' elseif check_trank && tranks(jj)==0 */
        /* 'weighted_Laplacian_tri_cleanmesh:68' us_smooth(jj,:) = 0; */
        for (i3 = 0; i3 < 3; i3++) {
          us_smooth->data[b_jj + us_smooth->size[0] * i3] = 0.0;
        }
      }
    }

    /* 'weighted_Laplacian_tri_cleanmesh:71' if any(ns_constrained(:,jj)~=0) */
    for (i3 = 0; i3 < 3; i3++) {
      b_ns_constrained[i3] = (ns_constrained->data[i3 + ns_constrained->size[0] *
        b_jj] != 0.0);
    }

    if (any(b_ns_constrained)) {
      /* 'weighted_Laplacian_tri_cleanmesh:72' nrm = ns_constrained(:,jj); */
      /* 'weighted_Laplacian_tri_cleanmesh:73' us_smooth(jj,:) = us_smooth(jj,:) - (us_smooth(jj,:)*nrm) * nrm'; */
      w = 0.0;
      ix = 0;
      nv = 0;
      for (k = 0; k < 3; k++) {
        w += us_smooth->data[b_jj + us_smooth->size[0] * ix] *
          ns_constrained->data[nv + ns_constrained->size[0] * b_jj];
        ix++;
        nv++;
      }

      for (i3 = 0; i3 < 3; i3++) {
        us_smooth->data[b_jj + us_smooth->size[0] * i3] -= w *
          ns_constrained->data[i3 + ns_constrained->size[0] * b_jj];
      }
    } else {
      /* 'weighted_Laplacian_tri_cleanmesh:74' else */
      /* 'weighted_Laplacian_tri_cleanmesh:75' nrm = nrm_surf(jj,:)'; */
      for (i3 = 0; i3 < 3; i3++) {
        nrm[i3] = fcenters->data[b_jj + fcenters->size[0] * i3];
      }

      /* 'weighted_Laplacian_tri_cleanmesh:77' us_smooth(jj,:) = us_smooth(jj,:) - ... */
      /* 'weighted_Laplacian_tri_cleanmesh:78'             (us_smooth(jj,:)*nrm) / (nrm'*nrm+1.e-100) * nrm'; */
      w = 0.0;
      ix = 0;
      nv = 0;
      for (k = 0; k < 3; k++) {
        w += us_smooth->data[b_jj + us_smooth->size[0] * ix] * nrm[nv];
        ix++;
        nv++;
      }

      y = 0.0;
      ix = 0;
      nv = 0;
      for (k = 0; k < 3; k++) {
        y += nrm[ix] * nrm[nv];
        ix++;
        nv++;
      }

      w /= y + 1.0E-100;
      for (i3 = 0; i3 < 3; i3++) {
        us_smooth->data[b_jj + us_smooth->size[0] * i3] -= w * nrm[i3];
      }
    }
  }

  emxFree_real_T(&ns_constrained);
  emxFree_real_T(&Vs);
  emxFree_int8_T(&tranks);
  emxFree_real_T(&fcenters);
}

/*
 * function alpha = check_prism( xs, us, ds)
 */
static real_T check_prism(const real_T xs[9], const real_T us[9])
{
  real_T alpha;
  int32_T ix;
  real_T u21[3];
  real_T u31[3];
  real_T x21[3];
  real_T x31[3];
  real_T c2[3];
  real_T b_u21[3];
  real_T b_x21[3];
  real_T a;
  int32_T b_ix;
  int32_T iy;
  real_T b;
  real_T c;
  int32_T b_iy;
  int32_T k;
  real_T max_abc;
  real_T u0;
  real_T u1;
  real_T s[2];

  /*  CHECK_PRISM: Determine whether a prism has positive or negative Jacobian */
  /*  determinant everywhere within a prism. */
  /*  */
  /*    ALPHA = CHECK_PRISM( XS, US) */
  /*    ALPHA = CHECK_PRISM( XS, US, DS) */
  /*    Returns smallest positive alpha for the prism to become invalid. */
  /*  */
  /*    XS, US, and DS are 3x3, and each row contains a point. */
  /*    XS is composed of coordinates of three vertices of bottom triangle. */
  /*    US is the displacement from bottom vertices to top vertices. */
  /*    DS is the reference directions used in the checking. */
  /* 'check_prism:14' assert( all(size(xs) == [3,3]) && all(size(us) == [3,3])); */
  /* 'check_prism:15' if nargin>2 */
  /* 'check_prism:17' u21 = us(2,1:3)-us(1,1:3); */
  for (ix = 0; ix < 3; ix++) {
    u21[ix] = us[1 + 3 * ix] - us[3 * ix];

    /* 'check_prism:17' u31 = us(3,1:3)-us(1,1:3); */
    u31[ix] = us[2 + 3 * ix] - us[3 * ix];

    /* 'check_prism:18' x21 = xs(2,1:3)-xs(1,1:3); */
    x21[ix] = xs[1 + 3 * ix] - xs[3 * ix];

    /* 'check_prism:18' x31 = xs(3,1:3)-xs(1,1:3); */
    x31[ix] = xs[2 + 3 * ix] - xs[3 * ix];
  }

  /* 'check_prism:20' c2 = cross_row(u21, u31); */
  /* CROSS_ROW Efficient computaiton of cross product between two  */
  /*  3-dimensional row vectors. */
  /*  CROSS_ROW(A,B) Efficiently computes the cross product between */
  /*  3-dimensional row vector A, and 3-dimensional row vector B. */
  /*  The result is a row vector. */
  /* 'cross_row:8' c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]; */
  c2[0] = u21[1] * u31[2] - u21[2] * u31[1];
  c2[1] = u21[2] * u31[0] - u21[0] * u31[2];
  c2[2] = u21[0] * u31[1] - u21[1] * u31[0];

  /* 'check_prism:21' c1 = cross_row(u21, x31)+cross_row(x21,u31); */
  /* CROSS_ROW Efficient computaiton of cross product between two  */
  /*  3-dimensional row vectors. */
  /*  CROSS_ROW(A,B) Efficiently computes the cross product between */
  /*  3-dimensional row vector A, and 3-dimensional row vector B. */
  /*  The result is a row vector. */
  /* 'cross_row:8' c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]; */
  /* CROSS_ROW Efficient computaiton of cross product between two  */
  /*  3-dimensional row vectors. */
  /*  CROSS_ROW(A,B) Efficiently computes the cross product between */
  /*  3-dimensional row vector A, and 3-dimensional row vector B. */
  /*  The result is a row vector. */
  /* 'cross_row:8' c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]; */
  b_u21[0] = u21[1] * x31[2] - u21[2] * x31[1];
  b_u21[1] = u21[2] * x31[0] - u21[0] * x31[2];
  b_u21[2] = u21[0] * x31[1] - u21[1] * x31[0];
  b_x21[0] = x21[1] * u31[2] - x21[2] * u31[1];
  b_x21[1] = x21[2] * u31[0] - x21[0] * u31[2];
  b_x21[2] = x21[0] * u31[1] - x21[1] * u31[0];

  /* 'check_prism:22' c0 = cross_row(x21, x31); */
  /* CROSS_ROW Efficient computaiton of cross product between two  */
  /*  3-dimensional row vectors. */
  /*  CROSS_ROW(A,B) Efficiently computes the cross product between */
  /*  3-dimensional row vector A, and 3-dimensional row vector B. */
  /*  The result is a row vector. */
  /* 'cross_row:8' c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]; */
  u21[0] = x21[1] * x31[2] - x21[2] * x31[1];
  u21[1] = x21[2] * x31[0] - x21[0] * x31[2];
  u21[2] = x21[0] * x31[1] - x21[1] * x31[0];

  /*  Check at vertices */
  /* 'check_prism:25' alpha = 1.e30; */
  alpha = 1.0E+30;

  /* 'check_prism:26' tol0  = -1.e-4; */
  /* 'check_prism:28' if nargin>2 */
  /* 'check_prism:47' a = c0*c2'; */
  a = 0.0;
  b_ix = 0;
  iy = 0;
  for (ix = 0; ix < 3; ix++) {
    u31[ix] = b_u21[ix] + b_x21[ix];
    a += u21[b_ix] * c2[iy];
    b_ix++;
    iy++;
  }

  /* 'check_prism:47' b = c0*c1'; */
  b = 0.0;
  b_ix = 0;
  iy = 0;

  /* 'check_prism:47' c = c0*c0'; */
  c = 0.0;
  ix = 0;
  b_iy = 0;
  for (k = 0; k < 3; k++) {
    b += u21[b_ix] * u31[iy];
    b_ix++;
    iy++;
    c += u21[ix] * u21[b_iy];
    ix++;
    b_iy++;
  }

  /* 'check_prism:48' s = solve_quadratic_eq(a, b, c); */
  /* SOLVE_QUADRATIC_EQ A numerically stable solver for quadratic equations. */
  /*  SOLVE_QUADRATIC_EQ( A, B, C) Solves for the roots of a quadratic  */
  /*  equation: Ax^2 + Bx + C = 0, where A, B, and C are the provided scalar  */
  /*  coefficients. The function outputs a 2x1 array of the function's roots.  */
  /*  If solution does not exist, the solution is set to a very large number,  */
  /*  (such as 1.e100). */
  /* 'solve_quadratic_eq:9' max_abc = max( abs(a), max(abs(b),abs(c))); */
  max_abc = fabs(a);
  u0 = fabs(b);
  u1 = fabs(c);
  u1 = u0 >= u1 ? u0 : u1;
  max_abc = max_abc >= u1 ? max_abc : u1;

  /* 'solve_quadratic_eq:11' if max_abc > 0 */
  if (max_abc > 0.0) {
    /*  Scale */
    /* 'solve_quadratic_eq:12' a = a / max_abc; */
    a /= max_abc;

    /* 'solve_quadratic_eq:12' b = b / max_abc; */
    b /= max_abc;

    /* 'solve_quadratic_eq:12' c = c / max_abc; */
    c /= max_abc;
  }

  /* 'solve_quadratic_eq:15' if ( a == 0) */
  if (a == 0.0) {
    /* 'solve_quadratic_eq:16' if ( b == 0) */
    if (b == 0.0) {
      /* 'solve_quadratic_eq:17' s = [1.e100; 1.e100]; */
      for (ix = 0; ix < 2; ix++) {
        s[ix] = 1.0E+100;
      }
    } else {
      /* 'solve_quadratic_eq:18' else */
      /* 'solve_quadratic_eq:19' s = [-c/b; 1.e100]; */
      s[0] = -c / b;
      s[1] = 1.0E+100;
    }
  } else {
    /* 'solve_quadratic_eq:21' else */
    /* 'solve_quadratic_eq:22' discr = b*b - 4*a*c; */
    max_abc = b * b - 4.0 * a * c;

    /*  Use 1.e-10 as tolerance against roundoff error */
    /* 'solve_quadratic_eq:25' if discr < -1.e-10 */
    if (max_abc < -1.0E-10) {
      /*  No solution */
      /* 'solve_quadratic_eq:26' s = [1.e100; 1.e100]; */
      for (ix = 0; ix < 2; ix++) {
        s[ix] = 1.0E+100;
      }
    } else if (fabs(max_abc) < 1.0E-10) {
      /* 'solve_quadratic_eq:27' elseif abs(discr) < 1.e-10 */
      /*  One solution */
      /* 'solve_quadratic_eq:28' s1 = -0.5*b / a; */
      max_abc = -0.5 * b / a;

      /* 'solve_quadratic_eq:29' s = [s1; s1]; */
      s[0] = max_abc;
      s[1] = max_abc;
    } else {
      /* 'solve_quadratic_eq:30' else */
      /*  Two solutions */
      /* 'solve_quadratic_eq:32' sqrt_d = sqrt( discr); */
      max_abc = sqrt(max_abc);

      /* 'solve_quadratic_eq:34' if ( b>0) */
      if (b > 0.0) {
        /* 'solve_quadratic_eq:35' temp = -b - sqrt_d; */
        max_abc = -b - max_abc;

        /* 'solve_quadratic_eq:36' s = [2*c/temp; 0.5*temp/a]; */
        s[0] = 2.0 * c / max_abc;
        s[1] = 0.5 * max_abc / a;
      } else {
        /* 'solve_quadratic_eq:37' else */
        /* 'solve_quadratic_eq:38' temp = -b + sqrt_d; */
        max_abc += -b;

        /* 'solve_quadratic_eq:39' s = [0.5*temp/a; 2*c/temp]; */
        s[0] = 0.5 * max_abc / a;
        s[1] = 2.0 * c / max_abc;
      }
    }
  }

  /* 'check_prism:50' if s(1)>=tol0 */
  if (s[0] >= -0.0001) {
    /* 'check_prism:50' alpha = min(alpha, max(0,s(1))); */
    u1 = s[0];
    u1 = 0.0 >= u1 ? 0.0 : u1;
    alpha = 1.0E+30 <= u1 ? 1.0E+30 : u1;
  }

  /* 'check_prism:51' if s(2)>=tol0 */
  if (s[1] >= -0.0001) {
    /* 'check_prism:51' alpha = min(alpha, max(0,s(2))); */
    u1 = s[1];
    u1 = 0.0 >= u1 ? 0.0 : u1;
    alpha = alpha <= u1 ? alpha : u1;
  }

  return alpha;
}

/*
 * function [ws,toocoarse] = compute_cmf_weights( pos, pnts, nrms, deg, interp, tol)
 */
static boolean_T compute_cmf_weights(const real_T pos[3], const emxArray_real_T *
  pnts, const emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws)
{
  boolean_T toocoarse;
  int32_T ix;
  int32_T j;
  real_T uu[3];
  real_T d;
  int32_T iy;
  int32_T k;
  real_T costheta;
  real_T d3;
  uint32_T bku;
  int32_T exitg1;

  /*  Compute weights for continuous moving frames. */
  /*  [ws,toocoarse] = compute_cmf_weights( pos, pnts, nrms, h, interp,tol) */
  /* 'compute_cmf_weights:5' MAXPNTS = 128; */
  /* 'compute_cmf_weights:6' coder.varsize( 'ws', [MAXPNTS,1], [1,0]); */
  /* 'compute_cmf_weights:7' assert( size(pnts,1)<=MAXPNTS); */
  /* 'compute_cmf_weights:9' if nargin<5 */
  /* 'compute_cmf_weights:11' if nargin<6 */
  /* 'compute_cmf_weights:17' epsilon = 1e-6; */
  /* 'compute_cmf_weights:18' toocoarse = false; */
  toocoarse = FALSE;

  /* 'compute_cmf_weights:19' ws = coder.nullcopy(zeros(size(pnts,1),1)); */
  ix = ws->size[0];
  ws->size[0] = pnts->size[0];
  emxEnsureCapacity((emxArray__common *)ws, ix, (int32_T)sizeof(real_T));

  /* 'compute_cmf_weights:20' for j = 1+int32(interp):int32(size(pnts,1)) */
  for (j = 0; j + 1 <= pnts->size[0]; j++) {
    /* 'compute_cmf_weights:21' uu = pnts(j,:)-pos; */
    for (ix = 0; ix < 3; ix++) {
      uu[ix] = pnts->data[j + pnts->size[0] * ix] - pos[ix];
    }

    /* 'compute_cmf_weights:23' d = uu*uu.'; */
    d = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      d += uu[ix] * uu[iy];
      ix++;
      iy++;
    }

    /* 'compute_cmf_weights:24' costheta = nrms(j,:)*nrms(1,:).'; */
    for (ix = 0; ix < 3; ix++) {
      uu[ix] = nrms->data[nrms->size[0] * ix];
    }

    costheta = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      costheta += nrms->data[j + nrms->size[0] * ix] * uu[iy];
      ix++;
      iy++;
    }

    /* 'compute_cmf_weights:26' if costheta>tol */
    if (costheta > 0.707106781186548) {
      /* 'compute_cmf_weights:27' ws(j-int32(interp)) = costheta/(sqrt(d+epsilon))^(deg/2); */
      d = sqrt(d + 1.0E-6);
      ix = 1;
      d3 = (real_T)deg / 2.0;
      d3 = d3 < 0.0 ? ceil(d3 - 0.5) : floor(d3 + 0.5);
      bku = (uint32_T)d3;
      do {
        exitg1 = 0U;
        if ((bku & 1U) != 0U) {
          d3 = d * (real_T)ix;
          if ((d3 < 4.503599627370496E+15) && (d3 > -4.503599627370496E+15)) {
            d3 = d3 < 0.0 ? ceil(d3 - 0.5) : floor(d3 + 0.5);
          }

          ix = (int32_T)d3;
        }

        bku >>= 1U;
        if ((int32_T)bku == 0) {
          exitg1 = 1U;
        } else {
          d *= d;
        }
      } while (exitg1 == 0U);

      d3 = costheta / (real_T)ix;
      if ((d3 < 4.503599627370496E+15) && (d3 > -4.503599627370496E+15)) {
        d3 = d3 < 0.0 ? ceil(d3 - 0.5) : floor(d3 + 0.5);
      }

      ws->data[j] = (real_T)(int32_T)d3;
    } else {
      /* 'compute_cmf_weights:28' else */
      /* 'compute_cmf_weights:29' toocoarse = true; */
      toocoarse = TRUE;
    }
  }

  return toocoarse;
}

/*
 * function [nrms, curs, prdirs] = compute_diffops_surf_cleanmesh(nv_clean,...
 *     xs, tris, nrms_proj, degree, ring, iterfit, nrms, curs, prdirs)
 */
static void b_compute_diffops_surf_cleanmesh(int32_T nv_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, const emxArray_real_T
  *nrms_proj, int32_T degree, real_T ring, emxArray_real_T *nrms, const
  emxArray_real_T *curs, const emxArray_real_T *prdirs)
{
  int32_T i10;
  uint32_T uv0[2];
  emxArray_int32_T *opphes;
  emxArray_int32_T *is_index;
  int32_T nv;
  int32_T ntris;
  int32_T ii;
  boolean_T exitg1;
  int32_T b_is_index[3];
  emxArray_int32_T *v2nv;
  emxArray_int32_T *v2he;
  int32_T ne;
  static const int8_T iv24[3] = { 1, 2, 0 };

  int32_T loop_ub;
  static const int8_T iv25[3] = { 2, 3, 1 };

  emxArray_real_T *b_curs;
  emxArray_real_T *b_prdirs;
  real_T u1;

  /* COMPUTE_DIFFOP_SURF_PARALLEL Compute differential operators on the */
  /* interior and boundary points of a submesh on a processor. */
  /* # coder.typeof( int32(0), [inf,3], [1,0]),coder.typeof( double(0), [inf,3], [1,0]), */
  /* # int32(0), double(0), true, coder.typeof( double(0), [inf,3], [1,0]), */
  /* # coder.typeof( double(0), [inf,2], [1,0]), */
  /* # coder.typeof( double(0), [inf,3], [1,0])} */
  /* 'compute_diffops_surf_cleanmesh:11' coder.inline('never') */
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
  for (i10 = 0; i10 < 2; i10++) {
    uv0[i10] = (uint32_T)tris->size[i10];
  }

  b_emxInit_int32_T(&opphes, 2);
  emxInit_int32_T(&is_index, 1);
  i10 = opphes->size[0] * opphes->size[1];
  opphes->size[0] = (int32_T)uv0[0];
  opphes->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)opphes, i10, (int32_T)sizeof(int32_T));

  /* 'compute_diffops_surf_cleanmesh:23' opphes = determine_opposite_halfedge_tri(int32(size(xs,1)), tris, opphes); */
  nv = xs->size[0];

  /* DETERMINE_OPPOSITE_HALFEDGE_TRI Determine opposite half-edges for triangle  */
  /* mesh. */
  /*  DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) Determines */
  /*  opposite half-edges for triangle mesh. The following explains the input */
  /*  and output arguments. */
  /*  */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS) */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) */
  /*  Computes mapping from each half-edge to its opposite half-edge for  */
  /*  triangle mesh. */
  /*  */
  /*  Convention: Each half-edge is indicated by <face_id,local_edge_id>. */
  /*  We assign 2 bits to local_edge_id (starts from 0). */
  /*  */
  /*  See also DETERMINE_OPPOSITE_HALFEDGE */
  /* coder.inline('never'); */
  /*  assert(isscalar(nv)&&isa(nv,'int32')); */
  /*  assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /* 'determine_opposite_halfedge_tri:23' nepE = int32(3); */
  /*  Number of edges per element */
  /* 'determine_opposite_halfedge_tri:24' next = int32([2,3,1]); */
  /* 'determine_opposite_halfedge_tri:25' inds = int32(1:3); */
  /* 'determine_opposite_halfedge_tri:27' ntris = int32(size(tris,1)); */
  ntris = tris->size[0];

  /* % First, build is_index to store starting position for each vertex. */
  /* 'determine_opposite_halfedge_tri:29' is_index = zeros(nv+1,1,'int32'); */
  i10 = is_index->size[0];
  is_index->size[0] = nv + 1;
  emxEnsureCapacity((emxArray__common *)is_index, i10, (int32_T)sizeof(int32_T));
  for (i10 = 0; i10 <= nv; i10++) {
    is_index->data[i10] = 0;
  }

  /* 'determine_opposite_halfedge_tri:30' for ii=1:ntris */
  ii = 0;
  exitg1 = 0U;
  while ((exitg1 == 0U) && (ii + 1 <= ntris)) {
    /* 'determine_opposite_halfedge_tri:31' if tris(ii,1)==0 */
    if (tris->data[ii] == 0) {
      /* 'determine_opposite_halfedge_tri:31' ntris=ii-1; */
      ntris = ii;
      exitg1 = 1U;
    } else {
      /* 'determine_opposite_halfedge_tri:32' is_index(tris(ii,inds)+1) = is_index(tris(ii,inds)+1) + 1; */
      for (i10 = 0; i10 < 3; i10++) {
        b_is_index[i10] = is_index->data[tris->data[ii + tris->size[0] * i10]] +
          1;
      }

      for (i10 = 0; i10 < 3; i10++) {
        is_index->data[tris->data[ii + tris->size[0] * i10]] = b_is_index[i10];
      }

      ii++;
    }
  }

  /* 'determine_opposite_halfedge_tri:34' is_index(1) = 1; */
  is_index->data[0] = 1;

  /* 'determine_opposite_halfedge_tri:35' for ii=1:nv */
  for (ii = 1; ii <= nv; ii++) {
    /* 'determine_opposite_halfedge_tri:36' is_index(ii+1) = is_index(ii) + is_index(ii+1); */
    is_index->data[ii] += is_index->data[ii - 1];
  }

  emxInit_int32_T(&v2nv, 1);
  emxInit_int32_T(&v2he, 1);

  /* 'determine_opposite_halfedge_tri:39' ne = ntris*nepE; */
  ne = ntris * 3;

  /* 'determine_opposite_halfedge_tri:40' v2nv = coder.nullcopy(zeros( ne,1, 'int32')); */
  i10 = v2nv->size[0];
  v2nv->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2nv, i10, (int32_T)sizeof(int32_T));

  /*  Vertex to next vertex in each halfedge. */
  /* 'determine_opposite_halfedge_tri:41' v2he = coder.nullcopy(zeros( ne,1, 'int32')); */
  i10 = v2he->size[0];
  v2he->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2he, i10, (int32_T)sizeof(int32_T));

  /*  Vertex to half-edge. */
  /* 'determine_opposite_halfedge_tri:42' for ii=1:ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'determine_opposite_halfedge_tri:43' v2nv(is_index( tris(ii,inds))) = tris(ii,next); */
    for (i10 = 0; i10 < 3; i10++) {
      v2nv->data[is_index->data[tris->data[ii + tris->size[0] * i10] - 1] - 1] =
        tris->data[ii + tris->size[0] * iv24[i10]];
    }

    /* 'determine_opposite_halfedge_tri:44' v2he(is_index( tris(ii,inds))) = 4*ii-1+inds; */
    ne = (ii + 1) << 2;
    for (i10 = 0; i10 < 3; i10++) {
      v2he->data[is_index->data[tris->data[ii + tris->size[0] * i10] - 1] - 1] =
        i10 + ne;
    }

    /* 'determine_opposite_halfedge_tri:45' is_index(tris(ii,inds)) = is_index(tris(ii,inds)) + 1; */
    for (i10 = 0; i10 < 3; i10++) {
      b_is_index[i10] = is_index->data[tris->data[ii + tris->size[0] * i10] - 1]
        + 1;
    }

    for (i10 = 0; i10 < 3; i10++) {
      is_index->data[tris->data[ii + tris->size[0] * i10] - 1] = b_is_index[i10];
    }
  }

  /* 'determine_opposite_halfedge_tri:47' for ii=nv-1:-1:1 */
  for (ii = nv - 1; ii > 0; ii--) {
    /* 'determine_opposite_halfedge_tri:47' is_index(ii+1) = is_index(ii); */
    is_index->data[ii] = is_index->data[ii - 1];
  }

  /* 'determine_opposite_halfedge_tri:48' is_index(1)=1; */
  is_index->data[0] = 1;

  /* % Set opphes */
  /* 'determine_opposite_halfedge_tri:50' if nargin<3 || isempty(opphes) */
  if (opphes->size[0] == 0) {
    /* 'determine_opposite_halfedge_tri:51' opphes = zeros(size(tris,1), nepE, 'int32'); */
    ne = tris->size[0];
    i10 = opphes->size[0] * opphes->size[1];
    opphes->size[0] = ne;
    opphes->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)opphes, i10, (int32_T)sizeof(int32_T));
    loop_ub = tris->size[0] * 3 - 1;
    for (i10 = 0; i10 <= loop_ub; i10++) {
      opphes->data[i10] = 0;
    }
  } else {
    /* 'determine_opposite_halfedge_tri:52' else */
    /* 'determine_opposite_halfedge_tri:53' assert( size(opphes,1)>=ntris && size(opphes,2)>=nepE); */
    /* 'determine_opposite_halfedge_tri:54' opphes(:) = 0; */
    i10 = opphes->size[0] * opphes->size[1];
    opphes->size[1] = 3;
    emxEnsureCapacity((emxArray__common *)opphes, i10, (int32_T)sizeof(int32_T));
    for (i10 = 0; i10 < 3; i10++) {
      loop_ub = opphes->size[0] - 1;
      for (ne = 0; ne <= loop_ub; ne++) {
        opphes->data[ne + opphes->size[0] * i10] = 0;
      }
    }
  }

  /* 'determine_opposite_halfedge_tri:57' for ii=1:ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'determine_opposite_halfedge_tri:58' for jj=int32(1):3 */
    for (ne = 0; ne < 3; ne++) {
      /* 'determine_opposite_halfedge_tri:59' if opphes(ii,jj) */
      if (opphes->data[ii + opphes->size[0] * ne] != 0) {
      } else {
        /* 'determine_opposite_halfedge_tri:60' v = tris(ii,jj); */
        /* 'determine_opposite_halfedge_tri:60' vn = tris(ii,next(jj)); */
        /*  LOCATE: Locate index col in v2nv(first:last) */
        /* 'determine_opposite_halfedge_tri:63' found = int32(0); */
        /* 'determine_opposite_halfedge_tri:64' for index = is_index(vn):is_index(vn+1)-1 */
        loop_ub = is_index->data[tris->data[ii + tris->size[0] * (iv25[ne] - 1)]]
          - 1;
        for (nv = is_index->data[tris->data[ii + tris->size[0] * (iv25[ne] - 1)]
             - 1] - 1; nv + 1 <= loop_ub; nv++) {
          /* 'determine_opposite_halfedge_tri:65' if v2nv(index)==v */
          if (v2nv->data[nv] == tris->data[ii + tris->size[0] * ne]) {
            /* 'determine_opposite_halfedge_tri:66' opp = v2he(index); */
            /* 'determine_opposite_halfedge_tri:67' opphes(ii,jj) = opp; */
            opphes->data[ii + opphes->size[0] * ne] = v2he->data[nv];

            /* opphes(heid2fid(opp),heid2leid(opp)) = ii*4+jj-1; */
            /* 'determine_opposite_halfedge_tri:69' opphes(bitshift(uint32(opp),-2),mod(opp,4)+1) = ii*4+jj-1; */
            opphes->data[((int32_T)((uint32_T)v2he->data[nv] >> 2U) +
                          opphes->size[0] * (v2he->data[nv] - ((v2he->data[nv] >>
              2) << 2))) - 1] = ((ii + 1) << 2) + ne;

            /* 'determine_opposite_halfedge_tri:71' found = found + 1; */
          }
        }

        /*  Check for consistency */
        /* 'determine_opposite_halfedge_tri:76' if found>1 */
      }
    }
  }

  emxFree_int32_T(&v2nv);
  emxFree_int32_T(&is_index);
  emxInit_real_T(&b_curs, 2);

  /*  Determine incident halfedge. */
  /* 'compute_diffops_surf_cleanmesh:26' v2he = coder.nullcopy(zeros( size(xs,1),1,'int32')); */
  i10 = v2he->size[0];
  v2he->size[0] = xs->size[0];
  emxEnsureCapacity((emxArray__common *)v2he, i10, (int32_T)sizeof(int32_T));

  /* 'compute_diffops_surf_cleanmesh:27' v2he = determine_incident_halfedges( tris, opphes, v2he); */
  b_determine_incident_halfedges(tris, opphes, v2he);

  /*  Invoke fitting algorithm. Do not use iterative fitting except for linear */
  /*  fitting. Do not use interp point. */
  /* 'compute_diffops_surf_cleanmesh:32' if nargin<8 && nargout<2 */
  /* 'compute_diffops_surf_cleanmesh:35' else */
  /* 'compute_diffops_surf_cleanmesh:36' [nrms,curs,prdirs] = polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, ... */
  /* 'compute_diffops_surf_cleanmesh:37'         nrms_proj, opphes, v2he, degree, ring, iterfit, true, nrms, curs, prdirs); */
  i10 = b_curs->size[0] * b_curs->size[1];
  b_curs->size[0] = curs->size[0];
  b_curs->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)b_curs, i10, (int32_T)sizeof(real_T));
  loop_ub = curs->size[0] * curs->size[1] - 1;
  for (i10 = 0; i10 <= loop_ub; i10++) {
    b_curs->data[i10] = curs->data[i10];
  }

  emxInit_real_T(&b_prdirs, 2);
  i10 = b_prdirs->size[0] * b_prdirs->size[1];
  b_prdirs->size[0] = prdirs->size[0];
  b_prdirs->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)b_prdirs, i10, (int32_T)sizeof(real_T));
  loop_ub = prdirs->size[0] * prdirs->size[1] - 1;
  for (i10 = 0; i10 <= loop_ub; i10++) {
    b_prdirs->data[i10] = prdirs->data[i10];
  }

  u1 = 3.5 <= ring ? 3.5 : ring;
  u1 = 1.0 >= u1 ? 1.0 : u1;
  polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, nrms_proj, opphes, v2he, degree,
    u1, nrms, b_curs, b_prdirs);
  emxFree_int32_T(&v2he);
  emxFree_real_T(&b_prdirs);
  emxFree_real_T(&b_curs);
  emxFree_int32_T(&opphes);
}

/*
 * function nrms = compute_hisurf_normals(nv_clean,xs,tris, degree)
 */
static void compute_hisurf_normals(int32_T nv_clean, const emxArray_real_T *xs,
  const emxArray_int32_T *tris, int32_T degree, emxArray_real_T *nrms)
{
  emxArray_real_T *r0;
  int32_T nv;
  int32_T i0;
  int32_T loop_ub;
  emxArray_real_T *nrms_proj;
  emxArray_real_T *r1;
  emxArray_real_T *r2;
  real_T d0;
  b_emxInit_real_T(&r0, 1);

  /* 'compute_hisurf_normals:3' coder.inline('never') */
  /* 'compute_hisurf_normals:4' ring = double((degree+1)/2); */
  /* 'compute_hisurf_normals:5' iterfit = false; */
  /* 'compute_hisurf_normals:6' flabel = zeros(size(tris,1),1); */
  /* 'compute_hisurf_normals:7' nv = size(xs,1); */
  nv = xs->size[0];

  /* 'compute_hisurf_normals:8' nrms = zeros(nv,3); */
  /* 'compute_hisurf_normals:8' curs = zeros(nv,2); */
  /* 'compute_hisurf_normals:8' prdirs = zeros(nv,3); */
  /* Step1:  Compute the averaged normals at the vertex */
  /* 'compute_hisurf_normals:11' nrms_proj = average_vertex_normal_tri_cleanmesh(nv_clean, xs, int32(tris), flabel); */
  i0 = r0->size[0];
  r0->size[0] = tris->size[0];
  emxEnsureCapacity((emxArray__common *)r0, i0, (int32_T)sizeof(real_T));
  loop_ub = tris->size[0] - 1;
  for (i0 = 0; i0 <= loop_ub; i0++) {
    r0->data[i0] = 0.0;
  }

  emxInit_real_T(&nrms_proj, 2);
  c_average_vertex_normal_tri_cle(nv_clean, xs, tris, r0, nrms_proj);

  /* Step2: Communicate variable "nrms_proj" at the ghost points (>nv_clean) */
  /* Step3: Compute normals from polynomial fitting */
  /* 'compute_hisurf_normals:16' [nrms] = compute_diffops_surf_cleanmesh(nv_clean, xs, int32(tris), ... */
  /* 'compute_hisurf_normals:17'     nrms_proj, int32(degree), ring, iterfit, nrms, curs, prdirs); */
  i0 = nrms->size[0] * nrms->size[1];
  nrms->size[0] = nv;
  nrms->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)nrms, i0, (int32_T)sizeof(real_T));
  emxFree_real_T(&r0);
  loop_ub = nv * 3 - 1;
  for (i0 = 0; i0 <= loop_ub; i0++) {
    nrms->data[i0] = 0.0;
  }

  emxInit_real_T(&r1, 2);
  i0 = r1->size[0] * r1->size[1];
  r1->size[0] = nv;
  r1->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)r1, i0, (int32_T)sizeof(real_T));
  loop_ub = (nv << 1) - 1;
  for (i0 = 0; i0 <= loop_ub; i0++) {
    r1->data[i0] = 0.0;
  }

  emxInit_real_T(&r2, 2);
  i0 = r2->size[0] * r2->size[1];
  r2->size[0] = nv;
  r2->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)r2, i0, (int32_T)sizeof(real_T));
  loop_ub = nv * 3 - 1;
  for (i0 = 0; i0 <= loop_ub; i0++) {
    r2->data[i0] = 0.0;
  }

  d0 = (real_T)(degree + 1) / 2.0;
  d0 = d0 < 0.0 ? ceil(d0 - 0.5) : floor(d0 + 0.5);
  b_compute_diffops_surf_cleanmesh(nv_clean, xs, tris, nrms_proj, degree, d0, nrms,
    r1, r2);

  /* Step4: (a) Update variable "nrms" of "nv_clean" pnts */
  /*  (b) Communicate variable "nrms" of ghost pnts (>nv_clean) */
  emxFree_real_T(&r2);
  emxFree_real_T(&r1);
  emxFree_real_T(&nrms_proj);
}

/*
 * function [As, bs, bs_lbl] = compute_medial_quadric_tri(xs, tris, flabel, ntri)
 */
static void compute_medial_quadric_tri(const emxArray_real_T *xs, const
  emxArray_int32_T *tris, const emxArray_int32_T *flabel, emxArray_real_T *As,
  emxArray_real_T *bs, emxArray_real_T *bs_lbl)
{
  int32_T nv;
  int32_T nume;
  int32_T ix;
  int32_T i6;
  int32_T jj;
  real_T b_xs[9];
  real_T c_xs[9];
  real_T a[3];
  real_T nrm_a[3];
  real_T nrm[3];
  real_T farea;
  int32_T i;
  real_T T[9];
  real_T b_As[2];
  emxArray_real_T *c_As;
  emxArray_real_T *d_As;
  emxArray_real_T *e_As;

  /* COMPUE_MEDIAL_QUADRIC_TRI   Compute medial quadric at each vertex. */
  /*  */
  /*  [AS, BS] = COMPUTE_MEDIAL_QUADRIC_TRI(XS, TRIS) */
  /*  [AS, BS] = COMPUTE_MEDIAL_QUADRIC_TRI(XS, TRIS, NTRI) computes the medial */
  /*     quadric and saves matrix and vector. XS is nx3 and TRIS is mx3. */
  /*     AS is 3x3xn and BS is 3xn. If NTRI is not present, it is set to nnz_elements(TRIS). */
  /*  */
  /*  [AS, BS, BS_LBL] = COMPUTE_MEDIAL_QUADRIC_TRI(XS, TRIS, FLABEL) */
  /*     computes medial quadric and consider faces with nonzero lablels */
  /*     as on constrained surfaces when computing BS_LBL. */
  /*  */
  /*   See also UPDATE_MEDIAL_QUADRIC_TRI, EIGENANALYSIS_SURF. */
  /* # coder.typeof(int32(0),[inf,1],[1,0]), int32(0)} */
  /* 'compute_medial_quadric_tri:18' assert( size(xs,2)==3 && size(tris,2)==3); */
  /* 'compute_medial_quadric_tri:19' assert( nargout<3 || nargin==3); */
  /* 'compute_medial_quadric_tri:21' nv = int32(size(xs,1)); */
  nv = xs->size[0];

  /* 'compute_medial_quadric_tri:22' if nargin<4 */
  /* 'compute_medial_quadric_tri:22' ntri = nnz_elements(tris); */
  /* NNZ_ELEMENTS the number of nonzero elements. */
  /*  NNZ_ELEMENTS(ELEMS) returns the number of elements with nonzero entries. */
  /*  Skip elements with zero entries. */
  /* 'nnz_elements:8' if size(elems,1)==0 || elems(1,1)==0 */
  if ((tris->size[0] == 0) || (tris->data[0] == 0)) {
    /* 'nnz_elements:9' ne = int32(0); */
    nume = 0;
  } else {
    /* 'nnz_elements:19' else */
    /* 'nnz_elements:20' nume = int32(size(elems,1)); */
    nume = tris->size[0];

    /* 'nnz_elements:22' for step=int32(size(elems,1)):-1:1 */
    ix = tris->size[0];
    while ((ix > 0) && (!(tris->data[ix - 1] != 0))) {
      /* 'nnz_elements:23' if elems(step,1) */
      /* 'nnz_elements:24' nume=nume-1; */
      nume--;
      ix--;
    }

    /* 'nnz_elements:26' ne=nume; */
  }

  /*  Compute normal tensor and bs */
  /* 'compute_medial_quadric_tri:25' As = zeros(3,3,nv); */
  i6 = As->size[0] * As->size[1] * As->size[2];
  As->size[0] = 3;
  As->size[1] = 3;
  As->size[2] = nv;
  emxEnsureCapacity((emxArray__common *)As, i6, (int32_T)sizeof(real_T));
  ix = 9 * nv - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    As->data[i6] = 0.0;
  }

  /* 'compute_medial_quadric_tri:26' bs = zeros(3,nv); */
  i6 = bs->size[0] * bs->size[1];
  bs->size[0] = 3;
  bs->size[1] = nv;
  emxEnsureCapacity((emxArray__common *)bs, i6, (int32_T)sizeof(real_T));
  ix = 3 * nv - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    bs->data[i6] = 0.0;
  }

  /*  Stores right-hand side for computing vertex displacements */
  /*  Allocate space */
  /* 'compute_medial_quadric_tri:29' if nargout>2 */
  /* 'compute_medial_quadric_tri:29' bs_lbl = zeros(3,nv); */
  i6 = bs_lbl->size[0] * bs_lbl->size[1];
  bs_lbl->size[0] = 3;
  bs_lbl->size[1] = nv;
  emxEnsureCapacity((emxArray__common *)bs_lbl, i6, (int32_T)sizeof(real_T));
  ix = 3 * nv - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    bs_lbl->data[i6] = 0.0;
  }

  /* 'compute_medial_quadric_tri:31' for jj=1:ntri */
  for (jj = 0; jj + 1 <= nume; jj++) {
    /* 'compute_medial_quadric_tri:32' vs = tris(jj,1:3); */
    /*  Compute face normals */
    /* 'compute_medial_quadric_tri:35' xs_tri = xs(vs,1:3); */
    /* 'compute_medial_quadric_tri:36' nrm = cross_col(xs_tri(2,1:3)-xs_tri(1,1:3),xs_tri(3,1:3)-xs_tri(1,1:3)); */
    for (i6 = 0; i6 < 3; i6++) {
      for (ix = 0; ix < 3; ix++) {
        b_xs[ix + 3 * i6] = xs->data[(tris->data[jj + tris->size[0] * ix] +
          xs->size[0] * i6) - 1];
      }
    }

    for (i6 = 0; i6 < 3; i6++) {
      for (ix = 0; ix < 3; ix++) {
        c_xs[ix + 3 * i6] = xs->data[(tris->data[jj + tris->size[0] * ix] +
          xs->size[0] * i6) - 1];
      }
    }

    for (i6 = 0; i6 < 3; i6++) {
      a[i6] = b_xs[1 + 3 * i6] - c_xs[3 * i6];
    }

    for (i6 = 0; i6 < 3; i6++) {
      for (ix = 0; ix < 3; ix++) {
        b_xs[ix + 3 * i6] = xs->data[(tris->data[jj + tris->size[0] * ix] +
          xs->size[0] * i6) - 1];
      }
    }

    for (i6 = 0; i6 < 3; i6++) {
      for (ix = 0; ix < 3; ix++) {
        c_xs[ix + 3 * i6] = xs->data[(tris->data[jj + tris->size[0] * ix] +
          xs->size[0] * i6) - 1];
      }
    }

    for (i6 = 0; i6 < 3; i6++) {
      nrm_a[i6] = b_xs[2 + 3 * i6] - c_xs[3 * i6];
    }

    /* CROSS_COL Efficient routine for computing cross product of two  */
    /* 3-dimensional column vectors. */
    /*  CROSS_COL(A,B) Efficiently computes the cross product between */
    /*  3-dimensional column vector A, and 3-dimensional column vector B. */
    /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
    nrm[0] = a[1] * nrm_a[2] - a[2] * nrm_a[1];
    nrm[1] = a[2] * nrm_a[0] - a[0] * nrm_a[2];
    nrm[2] = a[0] * nrm_a[1] - a[1] * nrm_a[0];

    /* 'compute_medial_quadric_tri:37' nrm_a = nrm; */
    /* 'compute_medial_quadric_tri:39' farea = sqrt(nrm'*nrm); */
    farea = 0.0;
    ix = 0;
    nv = 0;
    for (i = 0; i < 3; i++) {
      nrm_a[i] = nrm[i];
      farea += nrm[ix] * nrm[nv];
      ix++;
      nv++;
    }

    farea = sqrt(farea);

    /* 'compute_medial_quadric_tri:40' if farea==0 */
    /* 'compute_medial_quadric_tri:41' nrm = nrm / farea; */
    for (i6 = 0; i6 < 3; i6++) {
      nrm[i6] /= farea;
    }

    /*  Update As and bs for the vertices of the triangle. */
    /*  T=a * nrm * nrm', but update only upper-trangular part. */
    /* 'compute_medial_quadric_tri:45' T = [nrm(1)*nrm_a'; 0, nrm(2)*nrm_a(2:3)'; 0, 0, nrm(3)*nrm_a(3)]; */
    for (i6 = 0; i6 < 3; i6++) {
      T[3 * i6] = nrm[0] * nrm_a[i6];
    }

    T[1] = 0.0;
    for (i6 = 0; i6 < 2; i6++) {
      T[1 + 3 * (i6 + 1)] = nrm[1] * nrm_a[i6 + 1];
    }

    T[2] = 0.0;
    T[5] = 0.0;
    T[8] = nrm[2] * nrm_a[2];

    /* 'compute_medial_quadric_tri:47' for kk=int32(1):3 */
    for (nv = 0; nv < 3; nv++) {
      /* 'compute_medial_quadric_tri:48' v = tris(jj,kk); */
      /*  As(:,:,v) = As(:,:,v) + T, but update only upper-triangular part. */
      /* 'compute_medial_quadric_tri:51' As(1,:,v)   = As(1,:,v)+T(1,:); */
      ix = tris->data[jj + tris->size[0] * nv];
      i = tris->data[jj + tris->size[0] * nv];
      for (i6 = 0; i6 < 3; i6++) {
        a[i6] = As->data[As->size[0] * i6 + As->size[0] * As->size[1] * (i - 1)]
          + T[3 * i6];
      }

      for (i6 = 0; i6 < 3; i6++) {
        As->data[As->size[0] * i6 + As->size[0] * As->size[1] * (ix - 1)] = a[i6];
      }

      /* 'compute_medial_quadric_tri:52' As(2,2:3,v) = As(2,2:3,v)+T(2,2:3); */
      ix = tris->data[jj + tris->size[0] * nv];
      i = tris->data[jj + tris->size[0] * nv];
      for (i6 = 0; i6 < 2; i6++) {
        b_As[i6] = As->data[(As->size[0] * (1 + i6) + As->size[0] * As->size[1] *
                             (i - 1)) + 1] + T[1 + 3 * (1 + i6)];
      }

      for (i6 = 0; i6 < 2; i6++) {
        As->data[(As->size[0] * (1 + i6) + As->size[0] * As->size[1] * (ix - 1))
          + 1] = b_As[i6];
      }

      /* 'compute_medial_quadric_tri:53' As(3,3,v)   = As(3,3,v)+T(3,3); */
      As->data[((As->size[0] << 1) + As->size[0] * As->size[1] * (tris->data[jj
                 + tris->size[0] * nv] - 1)) + 2] += T[8];

      /* 'compute_medial_quadric_tri:55' bs(:,v) = bs(:,v) + nrm_a; */
      ix = tris->data[jj + tris->size[0] * nv];
      i = tris->data[jj + tris->size[0] * nv];
      for (i6 = 0; i6 < 3; i6++) {
        a[i6] = bs->data[i6 + bs->size[0] * (i - 1)] + nrm_a[i6];
      }

      for (i6 = 0; i6 < 3; i6++) {
        bs->data[i6 + bs->size[0] * (ix - 1)] = a[i6];
      }

      /*  Update bs_lbl */
      /* 'compute_medial_quadric_tri:58' if nargout>2 && size(flabel,1)>1 && flabel(jj)>0 */
      if ((flabel->size[0] > 1) && (flabel->data[jj] > 0)) {
        /* 'compute_medial_quadric_tri:59' bs_lbl(:,v) = bs_lbl(:,v) + nrm_a; */
        ix = tris->data[jj + tris->size[0] * nv];
        i = tris->data[jj + tris->size[0] * nv];
        for (i6 = 0; i6 < 3; i6++) {
          a[i6] = bs_lbl->data[i6 + bs_lbl->size[0] * (i - 1)] + nrm_a[i6];
        }

        for (i6 = 0; i6 < 3; i6++) {
          bs_lbl->data[i6 + bs_lbl->size[0] * (ix - 1)] = a[i6];
        }
      }
    }
  }

  c_emxInit_real_T(&c_As, 3);

  /*  Copy from upper-triangular to lower-triangular part of As. */
  /* 'compute_medial_quadric_tri:65' As(2,1,:) = As(1,2,:); */
  ix = As->size[2];
  i6 = c_As->size[0] * c_As->size[1] * c_As->size[2];
  c_As->size[0] = 1;
  c_As->size[1] = 1;
  c_As->size[2] = ix;
  emxEnsureCapacity((emxArray__common *)c_As, i6, (int32_T)sizeof(real_T));
  ix--;
  for (i6 = 0; i6 <= ix; i6++) {
    c_As->data[c_As->size[0] * c_As->size[1] * i6] = As->data[As->size[0] +
      As->size[0] * As->size[1] * i6];
  }

  ix = c_As->size[2] - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    As->data[1 + As->size[0] * As->size[1] * i6] = c_As->data[c_As->size[0] *
      c_As->size[1] * i6];
  }

  emxFree_real_T(&c_As);
  c_emxInit_real_T(&d_As, 3);

  /* 'compute_medial_quadric_tri:66' As(3,1,:) = As(1,3,:); */
  ix = As->size[2];
  i6 = d_As->size[0] * d_As->size[1] * d_As->size[2];
  d_As->size[0] = 1;
  d_As->size[1] = 1;
  d_As->size[2] = ix;
  emxEnsureCapacity((emxArray__common *)d_As, i6, (int32_T)sizeof(real_T));
  ix--;
  for (i6 = 0; i6 <= ix; i6++) {
    d_As->data[d_As->size[0] * d_As->size[1] * i6] = As->data[(As->size[0] << 1)
      + As->size[0] * As->size[1] * i6];
  }

  ix = d_As->size[2] - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    As->data[2 + As->size[0] * As->size[1] * i6] = d_As->data[d_As->size[0] *
      d_As->size[1] * i6];
  }

  emxFree_real_T(&d_As);
  c_emxInit_real_T(&e_As, 3);

  /* 'compute_medial_quadric_tri:67' As(3,2,:) = As(2,3,:); */
  ix = As->size[2];
  i6 = e_As->size[0] * e_As->size[1] * e_As->size[2];
  e_As->size[0] = 1;
  e_As->size[1] = 1;
  e_As->size[2] = ix;
  emxEnsureCapacity((emxArray__common *)e_As, i6, (int32_T)sizeof(real_T));
  ix--;
  for (i6 = 0; i6 <= ix; i6++) {
    e_As->data[e_As->size[0] * e_As->size[1] * i6] = As->data[((As->size[0] << 1)
      + As->size[0] * As->size[1] * i6) + 1];
  }

  ix = e_As->size[2] - 1;
  for (i6 = 0; i6 <= ix; i6++) {
    As->data[(As->size[0] + As->size[0] * As->size[1] * i6) + 2] = e_As->
      data[e_As->size[0] * e_As->size[1] * i6];
  }

  emxFree_real_T(&e_As);
}

/*
 * function bs = compute_qtb( Q, bs, ncols)
 */
static void compute_qtb(const emxArray_real_T *Q, emxArray_real_T *bs, int32_T
  ncols)
{
  int32_T nrow;
  int32_T k;
  real_T t2;
  int32_T ii;
  int32_T b_bs[2];
  emxArray_real_T c_bs;
  int32_T d_bs[2];
  int32_T e_bs[2];
  emxArray_real_T f_bs;

  /* 'compute_qtb:3' nrow = int32(size(Q,1)); */
  nrow = Q->size[0];

  /* 'compute_qtb:4' for k=1:ncols */
  for (k = 0; k + 1 <= ncols; k++) {
    /*  Optimized version for */
    /*  bs(k:nrow,:) = bs(k:nrow,:) - 2*v*(v'*bs(k:nrow,:)), */
    /*  where v is Q(k:npngs) */
    /* 'compute_qtb:8' for jj=1:int32(size(bs,2)) */
    /* 'compute_qtb:9' t2 = 0; */
    t2 = 0.0;

    /* 'compute_qtb:10' for ii=k:nrow */
    for (ii = k; ii + 1 <= nrow; ii++) {
      /* 'compute_qtb:10' t2 = t2+Q(ii,k)*bs(ii,jj); */
      b_bs[0] = bs->size[0];
      b_bs[1] = 1;
      c_bs = *bs;
      c_bs.size = (int32_T *)&b_bs;
      c_bs.numDimensions = 1;
      t2 += Q->data[ii + Q->size[0] * k] * c_bs.data[ii];
    }

    /* 'compute_qtb:11' t2 = t2+t2; */
    t2 += t2;

    /* 'compute_qtb:12' for ii=k:nrow */
    for (ii = k; ii + 1 <= nrow; ii++) {
      /* 'compute_qtb:12' bs(ii,jj) = bs(ii,jj) - t2 * Q(ii,k); */
      d_bs[0] = bs->size[0];
      d_bs[1] = 1;
      e_bs[0] = bs->size[0];
      e_bs[1] = 1;
      c_bs = *bs;
      c_bs.size = (int32_T *)&d_bs;
      c_bs.numDimensions = 1;
      f_bs = *bs;
      f_bs.size = (int32_T *)&e_bs;
      f_bs.numDimensions = 1;
      c_bs.data[ii] = f_bs.data[ii] - t2 * Q->data[ii + Q->size[0] * k];
    }
  }
}

/*
 * function [min_angle, max_angle, min_area, max_area] = compute_statistics_tris_global(nt_clean, xs, tris)
 */
static void compute_statistics_tris_global(int32_T nt_clean, const
  emxArray_real_T *xs, const emxArray_int32_T *tris, real_T *min_angle, real_T
  *max_angle, real_T *min_area, real_T *max_area)
{
  /* 'compute_statistics_tris_global:3' coder.inline('never') */
  /*  Step 1:  Compute the quality of the clean mesh for the current processor */
  /* 'compute_statistics_tris_global:5' [min_angle, max_angle, min_area, max_area] = compute_statistics_tris_cleanmesh(nt_clean, xs, tris); */
  c_compute_statistics_tris_clean(nt_clean, xs, tris, min_angle, max_angle,
    min_area, max_area);

  /*  Step 2: Obtain the global min_angle, max_angle, min_area, */
  /*  max_area . This step would require communicating the min_angle from other */
  /*  processor and performing a comparision among them to obtain the global */
  /*  minimum angle.  */
}

/*
 * function [ws,toocoarse] = compute_weights( us, nrms, deg, tol)
 */
static boolean_T compute_weights(const emxArray_real_T *us, const
  emxArray_real_T *nrms, int32_T deg, emxArray_real_T *ws)
{
  boolean_T toocoarse;
  boolean_T interp;
  int32_T ix;
  int32_T j;
  real_T b[2];
  real_T A;
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
  /* 'compute_weights:9' MAXPNTS = 128; */
  /* 'compute_weights:10' coder.varsize( 'ws', [MAXPNTS,1], [1,0]); */
  /* 'compute_weights:11' assert( size(us,1)<=MAXPNTS); */
  /* 'compute_weights:13' if nargin<4 || tol<0 */
  /* 'compute_weights:14' tol = 0; */
  /* 'compute_weights:17' interp = logical(size(nrms,1)-size(us,1)); */
  interp = (nrms->size[0] - us->size[0] != 0);

  /* 'compute_weights:18' epsilon = 1e-2; */
  /*  First, compute squared distance from each input point to the pos */
  /* 'compute_weights:21' ws = coder.nullcopy(zeros(size(us,1),1)); */
  ix = ws->size[0];
  ws->size[0] = us->size[0];
  emxEnsureCapacity((emxArray__common *)ws, ix, (int32_T)sizeof(real_T));

  /* 'compute_weights:22' for j = 1:int32(size(us,1)) */
  for (j = 0; j + 1 <= us->size[0]; j++) {
    /* 'compute_weights:23' ws(j) = us(j,:)*us(j,:)'; */
    for (ix = 0; ix < 2; ix++) {
      b[ix] = us->data[j + us->size[0] * ix];
    }

    A = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 2; k++) {
      A += us->data[j + us->size[0] * ix] * b[iy];
      ix++;
      iy++;
    }

    ws->data[j] = A;
  }

  /*  Second, compute a small correction term to guard aganst zero */
  /* 'compute_weights:27' h = sum(ws)/size(ws,1); */
  A = sum(ws);
  h = A / (real_T)ws->size[0];

  /*  Finally, compute the weights for each vertex */
  /* 'compute_weights:30' toocoarse = false; */
  toocoarse = FALSE;

  /* 'compute_weights:31' for j = 1:int32(size(us,1)) */
  for (j = 0; j + 1 <= us->size[0]; j++) {
    /* 'compute_weights:32' costheta = nrms(j+int32(interp),:)*nrms(1,:).'; */
    for (ix = 0; ix < 3; ix++) {
      b_b[ix] = nrms->data[nrms->size[0] * ix];
    }

    costheta = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      costheta += nrms->data[(j + interp) + nrms->size[0] * ix] * b_b[iy];
      ix++;
      iy++;
    }

    /* 'compute_weights:34' if costheta>tol */
    if (costheta > 0.0) {
      /* 'compute_weights:35' ws(j) = costheta*(ws(j)/h+epsilon)^(-double(deg)/2); */
      A = ws->data[j] / h + 0.01;
      u1 = -(real_T)deg / 2.0;
      A = pow(A, u1);
      ws->data[j] = costheta * A;
    } else {
      /* 'compute_weights:36' else */
      /* 'compute_weights:37' ws(j) = 0; */
      ws->data[j] = 0.0;

      /* 'compute_weights:38' toocoarse = true; */
      toocoarse = TRUE;
    }
  }

  return toocoarse;
}

/*
 * function cosa = cos_angle( ts1, ts2)
 */
static real_T cos_angle(const real_T ts1[3], const real_T ts2[3])
{
  real_T y;
  int32_T ix;
  int32_T iy;
  real_T b_y;
  int32_T b_ix;
  int32_T b_iy;
  real_T c_y;
  int32_T c_ix;
  int32_T c_iy;
  int32_T k;

  /*  subfunction for computing cot of an angle. */
  /* 'compute_statistics_tris_cleanmesh:52' cosa = (ts1'*ts2)/sqrt((ts1'*ts1)*(ts2'*ts2)+1.e-100); */
  y = 0.0;
  ix = 0;
  iy = 0;
  b_y = 0.0;
  b_ix = 0;
  b_iy = 0;
  c_y = 0.0;
  c_ix = 0;
  c_iy = 0;
  for (k = 0; k < 3; k++) {
    y += ts1[ix] * ts2[iy];
    ix++;
    iy++;
    b_y += ts1[b_ix] * ts1[b_iy];
    b_ix++;
    b_iy++;
    c_y += ts2[c_ix] * ts2[c_iy];
    c_ix++;
    c_iy++;
  }

  return y / sqrt(b_y * c_y + 1.0E-100);
}

/*
 *
 */
static boolean_T d_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv3[8] = { 'W', 'A', 'L', 'F', '_', 'N', 'R', 'M' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(7 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv3[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
}

static void d_emxInit_real_T(emxArray_real_T **pEmxArray, int32_T numDimensions)
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
 * function v2he = determine_incident_halfedges(elems, opphes, v2he)
 */
static void c_determine_incident_halfedges(const emxArray_int32_T *elems, const
  emxArray_int32_T *opphes, emxArray_int32_T *v2he)
{
  int32_T nv;
  int32_T ii;
  int32_T jj;
  boolean_T guard1 = FALSE;
  uint32_T a;

  /* DETERMINE_INCIDENT_HALFEDGES Determine an incident halfedges. */
  /*  DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) Determines incident */
  /*  halfedges of each vertex for a triangular, quadrilateral, or mixed mesh.  */
  /*  It gives higher priorities to border edges. The following explains inputs */
  /*  and outputs. */
  /*  */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*  V2HE = DETERMINE_INCIDENT_HALFEDGES(ELEMS,OPPHES,V2HE) */
  /*      ELEMS is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      OPPHES is mx3 (for triangle mesh) or mx4 (for quadrilateral mesh). */
  /*      V2HE is an array of size equal to number of vertices. */
  /*           It is passed by reference. */
  /*  */
  /*  See also DETERMINE_INCIDENT_HALFFACES, DETERMINE_INCIDENT_HALFVERTS */
  /* 'determine_incident_halfedges:18' coder.inline('never'); */
  /*  assert((size(elems,2)==3) && (size(elems,1)>=1) && isa(elems,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /*  assert((size(v2he,2)==1) && (size(v2he,1)>=1) && isa(v2he,'int32')); */
  /* 'determine_incident_halfedges:23' if nargin<3 */
  /*  Set nv to maximum value in elements */
  /* 'determine_incident_halfedges:25' nv = int32(0); */
  nv = 0;

  /* 'determine_incident_halfedges:26' for ii=1:int32(size(elems,1)) */
  ii = 0;
  while ((ii + 1 <= elems->size[0]) && (!(elems->data[ii] == 0))) {
    /* 'determine_incident_halfedges:27' if elems(ii,1)==0 */
    /* 'determine_incident_halfedges:29' for jj=1:int32(size(elems,2)) */
    for (jj = 0; jj < 3; jj++) {
      /* 'determine_incident_halfedges:30' if elems(ii,jj)>nv */
      if (elems->data[ii + elems->size[0] * jj] > nv) {
        /* 'determine_incident_halfedges:30' nv = elems(ii,jj); */
        nv = elems->data[ii + elems->size[0] * jj];
      }
    }

    ii++;
  }

  /* 'determine_incident_halfedges:34' v2he = zeros( nv, 1, 'int32'); */
  jj = v2he->size[0];
  v2he->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)v2he, jj, (int32_T)sizeof(int32_T));
  ii = nv - 1;
  for (jj = 0; jj <= ii; jj++) {
    v2he->data[jj] = 0;
  }

  /* 'determine_incident_halfedges:39' for kk=1:int32(size(elems,1)) */
  ii = 0;
  while ((ii + 1 <= elems->size[0]) && (!(elems->data[ii] == 0))) {
    /* 'determine_incident_halfedges:40' if elems(kk,1)==0 */
    /* 'determine_incident_halfedges:42' for lid=1:int32(size(elems,2)) */
    for (jj = 0; jj < 3; jj++) {
      /* 'determine_incident_halfedges:43' v = elems(kk,lid); */
      /* 'determine_incident_halfedges:44' if v>0 && (v2he(v)==0 || opphes( kk,lid) == 0 || ... */
      /* 'determine_incident_halfedges:45' 	     (opphes( int32( bitshift( uint32(v2he(v)),-2)), mod(v2he(v),4)+1) && opphes( kk, lid)<0)) */
      if (elems->data[ii + elems->size[0] * jj] > 0) {
        guard1 = FALSE;
        if ((v2he->data[elems->data[ii + elems->size[0] * jj] - 1] == 0) ||
            (opphes->data[ii + opphes->size[0] * jj] == 0)) {
          guard1 = TRUE;
        } else {
          a = (uint32_T)v2he->data[elems->data[ii + elems->size[0] * jj] - 1];
          if ((opphes->data[((int32_T)(a >> 2U) + opphes->size[0] * (v2he->
                 data[elems->data[ii + elems->size[0] * jj] - 1] - ((v2he->
                   data[elems->data[ii + elems->size[0] * jj] - 1] >> 2) << 2)))
               - 1] != 0) && (opphes->data[ii + opphes->size[0] * jj] < 0)) {
            guard1 = TRUE;
          }
        }

        if (guard1 == TRUE) {
          /* 'determine_incident_halfedges:46' v2he(v) = 4*kk + lid - 1; */
          v2he->data[elems->data[ii + elems->size[0] * jj] - 1] = ((ii + 1) << 2)
            + jj;
        }
      }
    }

    ii++;
  }
}

/*
 * function opphes = determine_opposite_halfedge(nv, elems, opphes)
 */
static void determine_opposite_halfedge(int32_T nv, const emxArray_int32_T
  *elems, emxArray_int32_T *opphes)
{
  emxArray_int32_T *is_index;
  int32_T ntris;
  int32_T found;
  int32_T ii;
  boolean_T exitg4;
  int32_T b_is_index[3];
  emxArray_int32_T *v2nv;
  emxArray_int32_T *v2he;
  int32_T ne;
  static const int8_T iv12[3] = { 1, 2, 0 };

  int32_T loop_ub;
  boolean_T exitg3;
  int32_T exitg2;
  boolean_T guard1 = FALSE;
  static const int8_T iv13[3] = { 2, 3, 1 };

  int32_T b_index;
  int32_T exitg1;
  boolean_T guard2 = FALSE;
  uint32_T a;
  emxInit_int32_T(&is_index, 1);

  /* DETERMINE_OPPOSITE_HALFEDGE determines the opposite half-edge of  */
  /*  each halfedge for an oriented, manifold surface mesh with or */
  /*  without boundary. It works for both triangle and quadrilateral  */
  /*  meshes that are either linear and quadratic. */
  /*  */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE(NV,ELEMS) */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE(NV,ELEMS,OPPHES) */
  /*  computes mapping from each half-edge to its opposite half-edge. This  */
  /*  function supports triangular, quadrilateral, and mixed meshes. */
  /*  */
  /*  Convention: Each half-edge is indicated by <face_id,local_edge_id>. */
  /*     We assign 2 bits to local_edge_id. */
  /*  */
  /*  See also DETERMINE_NEXTPAGE_SURF, DETERMINE_INCIDENT_HALFEDGES */
  /* 'determine_opposite_halfedge:16' coder.inline('never') */
  /* 'determine_opposite_halfedge:18' if nargin<3 */
  /* 'determine_opposite_halfedge:19' switch size(elems,2) */
  /* 'determine_opposite_halfedge:20' case {3,6} % tri */
  /*  tri */
  /* 'determine_opposite_halfedge:21' opphes = determine_opposite_halfedge_tri(nv, elems); */
  /* DETERMINE_OPPOSITE_HALFEDGE_TRI Determine opposite half-edges for triangle  */
  /* mesh. */
  /*  DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) Determines */
  /*  opposite half-edges for triangle mesh. The following explains the input */
  /*  and output arguments. */
  /*  */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS) */
  /*  OPPHES = DETERMINE_OPPOSITE_HALFEDGE_TRI(NV,TRIS,OPPHES) */
  /*  Computes mapping from each half-edge to its opposite half-edge for  */
  /*  triangle mesh. */
  /*  */
  /*  Convention: Each half-edge is indicated by <face_id,local_edge_id>. */
  /*  We assign 2 bits to local_edge_id (starts from 0). */
  /*  */
  /*  See also DETERMINE_OPPOSITE_HALFEDGE */
  /* coder.inline('never'); */
  /*  assert(isscalar(nv)&&isa(nv,'int32')); */
  /*  assert((size(tris,2)==3) && (size(tris,1)>=1) && isa(tris,'int32')); */
  /*  assert((size(opphes,2)==3) && (size(opphes,1)>=1) && isa(opphes,'int32')); */
  /* 'determine_opposite_halfedge_tri:23' nepE = int32(3); */
  /*  Number of edges per element */
  /* 'determine_opposite_halfedge_tri:24' next = int32([2,3,1]); */
  /* 'determine_opposite_halfedge_tri:25' inds = int32(1:3); */
  /* 'determine_opposite_halfedge_tri:27' ntris = int32(size(tris,1)); */
  ntris = elems->size[0];

  /* % First, build is_index to store starting position for each vertex. */
  /* 'determine_opposite_halfedge_tri:29' is_index = zeros(nv+1,1,'int32'); */
  found = is_index->size[0];
  is_index->size[0] = nv + 1;
  emxEnsureCapacity((emxArray__common *)is_index, found, (int32_T)sizeof(int32_T));
  for (found = 0; found <= nv; found++) {
    is_index->data[found] = 0;
  }

  /* 'determine_opposite_halfedge_tri:30' for ii=1:ntris */
  ii = 0;
  exitg4 = 0U;
  while ((exitg4 == 0U) && (ii + 1 <= ntris)) {
    /* 'determine_opposite_halfedge_tri:31' if tris(ii,1)==0 */
    if (elems->data[ii] == 0) {
      /* 'determine_opposite_halfedge_tri:31' ntris=ii-1; */
      ntris = ii;
      exitg4 = 1U;
    } else {
      /* 'determine_opposite_halfedge_tri:32' is_index(tris(ii,inds)+1) = is_index(tris(ii,inds)+1) + 1; */
      for (found = 0; found < 3; found++) {
        b_is_index[found] = is_index->data[elems->data[ii + elems->size[0] *
          found]] + 1;
      }

      for (found = 0; found < 3; found++) {
        is_index->data[elems->data[ii + elems->size[0] * found]] =
          b_is_index[found];
      }

      ii++;
    }
  }

  /* 'determine_opposite_halfedge_tri:34' is_index(1) = 1; */
  is_index->data[0] = 1;

  /* 'determine_opposite_halfedge_tri:35' for ii=1:nv */
  for (ii = 1; ii <= nv; ii++) {
    /* 'determine_opposite_halfedge_tri:36' is_index(ii+1) = is_index(ii) + is_index(ii+1); */
    is_index->data[ii] += is_index->data[ii - 1];
  }

  emxInit_int32_T(&v2nv, 1);
  emxInit_int32_T(&v2he, 1);

  /* 'determine_opposite_halfedge_tri:39' ne = ntris*nepE; */
  ne = ntris * 3;

  /* 'determine_opposite_halfedge_tri:40' v2nv = coder.nullcopy(zeros( ne,1, 'int32')); */
  found = v2nv->size[0];
  v2nv->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2nv, found, (int32_T)sizeof(int32_T));

  /*  Vertex to next vertex in each halfedge. */
  /* 'determine_opposite_halfedge_tri:41' v2he = coder.nullcopy(zeros( ne,1, 'int32')); */
  found = v2he->size[0];
  v2he->size[0] = ne;
  emxEnsureCapacity((emxArray__common *)v2he, found, (int32_T)sizeof(int32_T));

  /*  Vertex to half-edge. */
  /* 'determine_opposite_halfedge_tri:42' for ii=1:ntris */
  for (ii = 0; ii + 1 <= ntris; ii++) {
    /* 'determine_opposite_halfedge_tri:43' v2nv(is_index( tris(ii,inds))) = tris(ii,next); */
    for (found = 0; found < 3; found++) {
      v2nv->data[is_index->data[elems->data[ii + elems->size[0] * found] - 1] -
        1] = elems->data[ii + elems->size[0] * iv12[found]];
    }

    /* 'determine_opposite_halfedge_tri:44' v2he(is_index( tris(ii,inds))) = 4*ii-1+inds; */
    ne = (ii + 1) << 2;
    for (found = 0; found < 3; found++) {
      v2he->data[is_index->data[elems->data[ii + elems->size[0] * found] - 1] -
        1] = found + ne;
    }

    /* 'determine_opposite_halfedge_tri:45' is_index(tris(ii,inds)) = is_index(tris(ii,inds)) + 1; */
    for (found = 0; found < 3; found++) {
      b_is_index[found] = is_index->data[elems->data[ii + elems->size[0] * found]
        - 1] + 1;
    }

    for (found = 0; found < 3; found++) {
      is_index->data[elems->data[ii + elems->size[0] * found] - 1] =
        b_is_index[found];
    }
  }

  /* 'determine_opposite_halfedge_tri:47' for ii=nv-1:-1:1 */
  for (ii = nv - 1; ii > 0; ii--) {
    /* 'determine_opposite_halfedge_tri:47' is_index(ii+1) = is_index(ii); */
    is_index->data[ii] = is_index->data[ii - 1];
  }

  /* 'determine_opposite_halfedge_tri:48' is_index(1)=1; */
  is_index->data[0] = 1;

  /* % Set opphes */
  /* 'determine_opposite_halfedge_tri:50' if nargin<3 || isempty(opphes) */
  /* 'determine_opposite_halfedge_tri:51' opphes = zeros(size(tris,1), nepE, 'int32'); */
  ne = elems->size[0];
  found = opphes->size[0] * opphes->size[1];
  opphes->size[0] = ne;
  opphes->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)opphes, found, (int32_T)sizeof(int32_T));
  loop_ub = elems->size[0] * 3 - 1;
  for (found = 0; found <= loop_ub; found++) {
    opphes->data[found] = 0;
  }

  /* 'determine_opposite_halfedge_tri:57' for ii=1:ntris */
  ii = 0;
  exitg3 = 0U;
  while ((exitg3 == 0U) && (ii + 1 <= ntris)) {
    /* 'determine_opposite_halfedge_tri:58' for jj=int32(1):3 */
    ne = 0;
    do {
      exitg2 = 0U;
      if (ne + 1 < 4) {
        /* 'determine_opposite_halfedge_tri:59' if opphes(ii,jj) */
        guard1 = FALSE;
        if (opphes->data[ii + opphes->size[0] * ne] != 0) {
          guard1 = TRUE;
        } else {
          /* 'determine_opposite_halfedge_tri:60' v = tris(ii,jj); */
          /* 'determine_opposite_halfedge_tri:60' vn = tris(ii,next(jj)); */
          /*  LOCATE: Locate index col in v2nv(first:last) */
          /* 'determine_opposite_halfedge_tri:63' found = int32(0); */
          found = 0;

          /* 'determine_opposite_halfedge_tri:64' for index = is_index(vn):is_index(vn+1)-1 */
          loop_ub = is_index->data[elems->data[ii + elems->size[0] * (iv13[ne] -
            1)]] - 1;
          for (b_index = is_index->data[elems->data[ii + elems->size[0] *
               (iv13[ne] - 1)] - 1] - 1; b_index + 1 <= loop_ub; b_index++) {
            /* 'determine_opposite_halfedge_tri:65' if v2nv(index)==v */
            if (v2nv->data[b_index] == elems->data[ii + elems->size[0] * ne]) {
              /* 'determine_opposite_halfedge_tri:66' opp = v2he(index); */
              /* 'determine_opposite_halfedge_tri:67' opphes(ii,jj) = opp; */
              opphes->data[ii + opphes->size[0] * ne] = v2he->data[b_index];

              /* opphes(heid2fid(opp),heid2leid(opp)) = ii*4+jj-1; */
              /* 'determine_opposite_halfedge_tri:69' opphes(bitshift(uint32(opp),-2),mod(opp,4)+1) = ii*4+jj-1; */
              opphes->data[((int32_T)((uint32_T)v2he->data[b_index] >> 2U) +
                            opphes->size[0] * (v2he->data[b_index] -
                ((v2he->data[b_index] >> 2) << 2))) - 1] = ((ii + 1) << 2) + ne;

              /* 'determine_opposite_halfedge_tri:71' found = found + 1; */
              found++;
            }
          }

          /*  Check for consistency */
          /* 'determine_opposite_halfedge_tri:76' if found>1 */
          if ((found > 1) || (found != 0)) {
            /* 'determine_opposite_halfedge_tri:77' error( 'Input mesh is not an oriented manifold.'); */
            guard1 = TRUE;
          } else {
            /* 'determine_opposite_halfedge_tri:78' elseif ~found */
            /* 'determine_opposite_halfedge_tri:79' for index = is_index(v):is_index(v+1)-1 */
            loop_ub = is_index->data[elems->data[ii + elems->size[0] * ne]] - 1;
            b_index = is_index->data[elems->data[ii + elems->size[0] * ne] - 1];
            do {
              exitg1 = 0U;
              if (b_index <= loop_ub) {
                /* 'determine_opposite_halfedge_tri:80' if v2nv(index)==vn && int32(bitshift( uint32(v2he(index)),-2))~=ii */
                guard2 = FALSE;
                if (v2nv->data[b_index - 1] == elems->data[ii + elems->size[0] *
                    (iv13[ne] - 1)]) {
                  a = (uint32_T)v2he->data[b_index - 1];
                  if ((int32_T)(a >> 2U) != ii + 1) {
                    /* 'determine_opposite_halfedge_tri:81' if nargin==3 */
                    /* 'determine_opposite_halfedge_tri:83' else */
                    /* 'determine_opposite_halfedge_tri:84' opphes = zeros(0,3, 'int32'); */
                    found = opphes->size[0] * opphes->size[1];
                    opphes->size[0] = 0;
                    opphes->size[1] = 3;
                    emxEnsureCapacity((emxArray__common *)opphes, found,
                                      (int32_T)sizeof(int32_T));
                    exitg1 = 2U;
                  } else {
                    guard2 = TRUE;
                  }
                } else {
                  guard2 = TRUE;
                }

                if (guard2 == TRUE) {
                  b_index++;
                }
              } else {
                exitg1 = 1U;
              }
            } while (exitg1 == 0U);

            if (exitg1 == 1U) {
              guard1 = TRUE;
            } else {
              exitg2 = 2U;
            }
          }
        }

        if (guard1 == TRUE) {
          ne++;
        }
      } else {
        ii++;
        exitg2 = 1U;
      }
    } while (exitg2 == 0U);

    if (exitg2 == 1U) {
    } else {
      exitg3 = 1U;
    }
  }

  emxFree_int32_T(&v2he);
  emxFree_int32_T(&v2nv);
  emxFree_int32_T(&is_index);
}

/*
 *
 */
static boolean_T e_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg1;
  static const char_T cv4[3] = { 'c', 'm', 'f' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg1 = 0U;
    if (k + 1 <= 3) {
      if (a[k] != cv4[k]) {
        exitg1 = 1U;
      } else {
        k++;
      }
    } else {
      b_bool = TRUE;
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  return b_bool;
}

/*
 * function [V,lambdas] = eig3_sorted(A)
 */
static void eig3_sorted(const real_T A[9], real_T V[9], real_T lambdas[3])
{
  int32_T ix;
  real_T b[2];
  real_T h;
  int32_T iy;
  int32_T k;
  real_T g;
  real_T u2;
  real_T omega;
  real_T f;
  real_T q2;
  real_T w[3];
  static const int8_T iv15[3] = { 1, 0, 0 };

  real_T e[3];
  int32_T exitg1;
  int32_T m;
  real_T p;
  int32_T j;
  real_T b_V[6];

  /* EIG3_SORTED Perform eigenvalue-decomposition of 3x3 matrix A */
  /*  [V,lambdas] = eig3_sorted(A) computes eigenvalues and eigenvectors of a  */
  /*  3x3 symmetric matrix A and stores them into 3x1 vector lambdas and 3x3  */
  /*  matrix V, respectively. The eigenvalues are sorted from large to small. */
  /*  */
  /*  The function accesses only the diagonal and upper triangular parts of A. */
  /*  The access is read-only. */
  /*  */
  /*  See also eig, eig2, eig3 */
  /* 'eig3_sorted:12' [V,D] = eig3(A); */
  /* EIG3 Computes eigenvalues and eigenvectors of 3x3 symmetric matrix. */
  /*  [Q,LAMBDAS] = EIG3(A) Computes the 3x3 eigenvector matrix Q, and the 3x3  */
  /*  diagonal eigenvalue matrix LAMBDAS, provided any 3x3 symmetric matrix A.  */
  /*  It has same I/O convention as EIG. */
  /*  */
  /*  See also EIG, EIG2, EIG3_SORTED */
  /*  Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3 */
  /*  matrix A using the QL algorithm with implicit shifts, preceded by a */
  /*  Householder reduction to tridiagonal form. */
  /*  The function accesses only the diagonal and upper triangular parts of A. */
  /*  The access is read-only. */
  /*  ---------------------------------------------------------------------------- */
  /*  Parameters: */
  /*    A: The symmetric input matrix */
  /*    Q: Storage buffer for eigenvectors */
  /*    lambdas: Storage buffer for eigenvalues */
  /*  ---------------------------------------------------------------------------- */
  /*  Note: Modified from implementation of Joachim Kopp (www.mpi-hd.mpg.de/~jkopp/3x3/) */
  /*  ------------------------------------------------------------------------- */
  /*  Transform A to real tridiagonal form by the Householder method */
  /* 'eig3:25' assert( isa(A,'double')); */
  /* 'eig3:27' [Q,w,e1] = dsytrd3(A); */
  /*  ---------------------------------------------------------------------------- */
  /*  Reduces a symmetric 3x3 matrix to tridiagonal form by applying */
  /*  (unitary) Householder transformations: */
  /*             [ d[0]  e[0]       ] */
  /*     A = Q . [ e[0]  d[1]  e[1] ] . Q^T */
  /*             [       e[1]  d[2] ] */
  /*  The function accesses only the diagonal and upper triangular */
  /*  parts of A. The access is read-only. */
  /*  Bring first row and column to the desired form */
  /* 'eig3:117' h = A(1,2:3)*A(1,2:3)'; */
  for (ix = 0; ix < 2; ix++) {
    b[ix] = A[3 * (1 + ix)];
  }

  h = 0.0;
  ix = 1;
  iy = 0;
  for (k = 0; k < 2; k++) {
    h += A[3 * ix] * b[iy];
    ix++;
    iy++;
  }

  /* 'eig3:119' if A(1,2) > 0 */
  if (A[3] > 0.0) {
    /* 'eig3:120' g = -sqrt(h); */
    g = -sqrt(h);
  } else {
    /* 'eig3:121' else */
    /* 'eig3:122' g = sqrt(h); */
    g = sqrt(h);
  }

  /* 'eig3:125' e = [g; A(2,3)]; */
  b[0] = g;
  b[1] = A[7];

  /* 'eig3:126' f = g * A(1,2); */
  /* 'eig3:127' u2 = A(1,2) - g; */
  u2 = A[3] - g;

  /* 'eig3:127' u3 = A(1,3); */
  /* 'eig3:129' omega = h - f; */
  omega = h - g * A[3];

  /* 'eig3:130' if omega > 0.0 */
  if (omega > 0.0) {
    /* 'eig3:131' omega = 1.0 / omega; */
    omega = 1.0 / omega;

    /* 'eig3:133' f    = A(2,2) * u2 + A(2,3) * u3; */
    f = A[4] * u2 + A[7] * A[6];

    /* 'eig3:134' q2 = omega * f; */
    q2 = omega * f;

    /*  p */
    /* 'eig3:135' K    = u2 * f; */
    h = u2 * f;

    /*  u* A u */
    /* 'eig3:137' f    = A(2,3) * u2 + A(3,3) * u3; */
    f = A[7] * u2 + A[8] * A[6];

    /* 'eig3:138' q3 = omega * f; */
    /*  p */
    /* 'eig3:139' K   = K + u3 * f; */
    h += A[6] * f;

    /*  u* A u */
    /* 'eig3:141' K   = 0.5 * K * omega * omega; */
    h = 0.5 * h * omega * omega;

    /* 'eig3:143' q2 = q2 - K * u2; */
    q2 -= h * u2;

    /* 'eig3:144' q3 = q3 - K * u3; */
    h = omega * f - h * A[6];

    /* 'eig3:145' d = [A(1,1); A(2,2)-2.0*q2*u2; A(3,3)-2.0*q3*u3]; */
    w[0] = A[0];
    w[1] = A[4] - 2.0 * q2 * u2;
    w[2] = A[8] - 2.0 * h * A[6];

    /*  Store inverse Householder transformation in Q */
    /* 'eig3:148' Q = [1, 0, 0; */
    /* 'eig3:149'         0, 1 - omega * u2 *u2, - omega * u2 *u3; */
    /* 'eig3:150'         0, 0, 1 - omega * u3 *u3]; */
    for (ix = 0; ix < 3; ix++) {
      V[3 * ix] = (real_T)iv15[ix];
    }

    V[1] = 0.0;
    V[4] = 1.0 - omega * u2 * u2;
    V[7] = -omega * u2 * A[6];
    V[2] = 0.0;
    V[5] = 0.0;
    V[8] = 1.0 - omega * A[6] * A[6];

    /* 'eig3:151' Q(3,2) = Q(2,3); */
    V[5] = V[7];

    /*  Calculate updated A(2,3) and store it in e(2) */
    /* 'eig3:154' e(2) = e(2) - q2*u3 - u2*q3; */
    b[1] = (b[1] - q2 * A[6]) - u2 * h;
  } else {
    /* 'eig3:155' else */
    /* 'eig3:156' Q = eye(3,class(A)); */
    for (ix = 0; ix < 9; ix++) {
      V[ix] = 0.0;
    }

    for (ix = 0; ix < 3; ix++) {
      V[ix + 3 * ix] = 1.0;
    }

    /* 'eig3:157' d = [A(1,1); A(2,2); A(3,3)]; */
    w[0] = A[0];
    w[1] = A[4];
    w[2] = A[8];
  }

  /* 'eig3:28' e = [e1(1); e1(2); 0]; */
  e[0] = b[0];
  e[1] = b[1];
  e[2] = 0.0;

  /*  Calculate eigensystem of the remaining real symmetric tridiagonal matrix */
  /*  with the QL method */
  /*  */
  /*  Loop over all off-diagonal elements */
  /* 'eig3:34' for k=int32(1):2 */
  for (k = 0; k < 2; k++) {
    /* 'eig3:35' nIter = int32(0); */
    iy = 0;

    /* 'eig3:36' while true */
    do {
      exitg1 = 0U;

      /*  Check for convergence and exit iteration loop if off-diagonal */
      /*  element e(k) is zero */
      /* 'eig3:39' g = abs(w(k))+abs(w(k+1)); */
      g = fabs(w[k]) + fabs(w[k + 1]);

      /* 'eig3:40' if (abs(e(k)) + g == g) */
      if (fabs(e[k]) + g == g) {
        exitg1 = 1U;
      } else {
        /* 'eig3:42' if k==1 */
        if (k + 1 == 1) {
          /* 'eig3:43' g = abs(w(2))+abs(w(3)); */
          g = fabs(w[1]) + fabs(w[2]);

          /* 'eig3:44' m = int32(3 - (abs(e(2)) + g == g)); */
          m = 2 - (fabs(e[1]) + g == g);
        } else {
          /* 'eig3:45' else */
          /* 'eig3:46' m = int32(3); */
          m = 2;
        }

        /* 'eig3:49' nIter = nIter + 1; */
        iy++;

        /* 'eig3:49' if (nIter >= 30) */
        if (iy >= 30) {
          exitg1 = 1U;
        } else {
          /*  Calculate g = d_m - k */
          /* 'eig3:52' g = (w(k+1) - w(k)) / (e(k) + e(k)); */
          g = (w[k + 1] - w[k]) / (e[k] + e[k]);

          /* 'eig3:53' r = sqrt(g*g + 1.0); */
          h = sqrt(g * g + 1.0);

          /* 'eig3:54' if (g > 0) */
          if (g > 0.0) {
            /* 'eig3:55' g = w(m) - w(k) + e(k)/(g + r); */
            g = (w[m] - w[k]) + e[k] / (g + h);
          } else {
            /* 'eig3:56' else */
            /* 'eig3:57' g = w(m) - w(k) + e(k)/(g - r); */
            g = (w[m] - w[k]) + e[k] / (g - h);
          }

          /* 'eig3:60' s = cast(1.0, class(A)); */
          omega = 1.0;

          /* 'eig3:60' c = cast(1.0, class(A)); */
          u2 = 1.0;

          /* 'eig3:61' p = cast(0.0, class(A)); */
          p = 0.0;

          /* 'eig3:62' for i=m-1:-1:k */
          for (ix = m; ix >= k + 1; ix--) {
            /* 'eig3:63' f = s * e(i); */
            f = omega * e[ix - 1];

            /* 'eig3:64' b = c * e(i); */
            q2 = u2 * e[ix - 1];

            /* 'eig3:65' if (abs(f) > abs(g)) */
            if (fabs(f) > fabs(g)) {
              /* 'eig3:66' c      = g / f; */
              u2 = g / f;

              /* 'eig3:67' r      = sqrt(c*c + 1.0); */
              h = sqrt(u2 * u2 + 1.0);

              /* 'eig3:68' e(i+1) = f * r; */
              e[ix] = f * h;

              /* 'eig3:69' s      = 1.0/r; */
              omega = 1.0 / h;

              /* 'eig3:70' c      = c * s; */
              u2 *= omega;
            } else {
              /* 'eig3:71' else */
              /* 'eig3:72' s      = f / g; */
              omega = f / g;

              /* 'eig3:73' r      = sqrt(s*s + 1.0); */
              h = sqrt(omega * omega + 1.0);

              /* 'eig3:74' e(i+1) = g * r; */
              e[ix] = g * h;

              /* 'eig3:75' c      = 1.0/r; */
              u2 = 1.0 / h;

              /* 'eig3:76' s      = s * c; */
              omega *= u2;
            }

            /* 'eig3:79' g = w(i+1) - p; */
            g = w[ix] - p;

            /* 'eig3:80' r = (w(i) - g)*s + 2.0*c*b; */
            h = (w[ix - 1] - g) * omega + 2.0 * u2 * q2;

            /* 'eig3:81' p = s * r; */
            p = omega * h;

            /* 'eig3:82' w(i+1) = g + p; */
            w[ix] = g + p;

            /* 'eig3:83' g = c*r - b; */
            g = u2 * h - q2;

            /*  Form eigenvectors */
            /* 'eig3:86' for j=int32(1):3 */
            for (j = 0; j < 3; j++) {
              /* 'eig3:87' t = Q(j,i+1); */
              h = V[j + 3 * ix];

              /* 'eig3:88' Q(j,i+1) = s*Q(j,i) + c*t; */
              V[j + 3 * ix] = omega * V[j + 3 * (ix - 1)] + u2 * V[j + 3 * ix];

              /* 'eig3:89' Q(j,i)   = c*Q(j,i) - s*t; */
              V[j + 3 * (ix - 1)] = u2 * V[j + 3 * (ix - 1)] - omega * h;
            }
          }

          /* 'eig3:92' w(k)  = w(k) - p; */
          w[k] -= p;

          /* 'eig3:93' e(k)  = g; */
          e[k] = g;

          /* 'eig3:94' e(m)  = 0.0; */
          e[m] = 0.0;
        }
      }
    } while (exitg1 == 0U);
  }

  /*  If only one output, then set first output argument to eigenvalues. */
  /*  Otherwise, set second output argument to eigenvalues. */
  /* 'eig3:100' if nargout<=1 */
  /* 'eig3:102' else */
  /* 'eig3:103' lambdas = [w(1), 0 0 ; 0 w(2) 0; 0 0 w(3)]; */
  /* 'eig3_sorted:13' lambdas = [D(1,1);D(2,2);D(3,3)]; */
  lambdas[0] = w[0];
  lambdas[1] = w[1];
  lambdas[2] = w[2];

  /* 'eig3_sorted:14' if lambdas(1)<lambdas(2) */
  if (lambdas[0] < lambdas[1]) {
    /* 'eig3_sorted:15' lambdas([1,2])=lambdas([2,1]); */
    for (ix = 0; ix < 2; ix++) {
      b[ix] = lambdas[1 - ix];
    }

    /* 'eig3_sorted:16' V(:,[1,2]) = V(:,[2,1]); */
    for (ix = 0; ix < 2; ix++) {
      lambdas[ix] = b[ix];
      for (iy = 0; iy < 3; iy++) {
        b_V[iy + 3 * ix] = V[iy + 3 * (1 - ix)];
      }
    }

    for (ix = 0; ix < 2; ix++) {
      for (iy = 0; iy < 3; iy++) {
        V[iy + 3 * ix] = b_V[iy + 3 * ix];
      }
    }
  }

  /* 'eig3_sorted:18' if lambdas(1)<lambdas(3) */
  if (lambdas[0] < lambdas[2]) {
    /* 'eig3_sorted:19' lambdas([1,3])=lambdas([3,1]); */
    for (ix = 0; ix < 2; ix++) {
      b[ix] = lambdas[2 + -2 * ix];
    }

    for (ix = 0; ix < 2; ix++) {
      lambdas[ix << 1] = b[ix];
    }

    /* 'eig3_sorted:20' V(:,[1,3]) = V(:,[3,1]); */
    for (ix = 0; ix < 2; ix++) {
      for (iy = 0; iy < 3; iy++) {
        b_V[iy + 3 * ix] = V[iy + 3 * (2 + -2 * ix)];
      }
    }

    for (ix = 0; ix < 2; ix++) {
      for (iy = 0; iy < 3; iy++) {
        V[iy + 3 * (ix << 1)] = b_V[iy + 3 * ix];
      }
    }
  }

  /* 'eig3_sorted:22' if lambdas(2)<lambdas(3) */
  if (lambdas[1] < lambdas[2]) {
    /* 'eig3_sorted:23' lambdas([2,3])=lambdas([3,2]); */
    for (ix = 0; ix < 2; ix++) {
      b[ix] = lambdas[2 - ix];
    }

    /* 'eig3_sorted:24' V(:,[2,3]) = V(:,[3,2]); */
    for (ix = 0; ix < 2; ix++) {
      lambdas[1 + ix] = b[ix];
      for (iy = 0; iy < 3; iy++) {
        b_V[iy + 3 * ix] = V[iy + 3 * (2 - ix)];
      }
    }

    for (ix = 0; ix < 2; ix++) {
      for (iy = 0; iy < 3; iy++) {
        V[iy + 3 * (1 + ix)] = b_V[iy + 3 * ix];
      }
    }
  }
}

/*
 * function [us, Vs, tranks, lambdas] = eigenanalysis_surf( As, bs, isridge, us, to_update)
 */
static void eigenanalysis_surf(const emxArray_real_T *As, const emxArray_real_T *
  bs, const emxArray_boolean_T *isridge, emxArray_real_T *us, emxArray_real_T
  *Vs, emxArray_int8_T *tranks)
{
  int32_T nv;
  int32_T ix;
  int32_T jj;
  int32_T iy;
  real_T b_As[9];
  real_T ls[3];
  real_T V[9];
  real_T a[3];
  real_T b_sign;
  int32_T k;
  real_T y;
  real_T d[3];
  int8_T trank;
  boolean_T guard1 = FALSE;

  /* EIGENANALYSIS_SURF   Performs eigenvalue decomposition of normal tensor. */
  /*   US = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS] = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS, TRANKS] = EIGENANALYSIS_SURF( AS, BS) */
  /*   [US, VS, TRANKS, LAMBDAS] = EIGENANALYSIS_SURF( AS, BS) */
  /*        solves normal displacement vector and saves into US. */
  /*        Eigenvectors and eignvalues are stored into VS and LAMBDAS. */
  /*        Tangent ranks are stored into TRANKS. */
  /*  */
  /*   EIGENANALYSIS_SURF( AS, BS, ISRIDGE) */
  /*        also considers the input flags for ridge vertices. */
  /*  */
  /*   EIGENANALYSIS_SURF( AS, BS, ISRIDGE, US, TO_UPDATE) */
  /*        updates us only for vertices with to_update set to a nonzero value. */
  /*  */
  /*   See also COMPUTE_MEDIAL_QUADRIC_SURF, COMPUTE_OFFSET_QUADRIC_SURF. */
  /* # coder.typeof(false,[inf,1],[1,0]),coder.typeof(0,[inf,3],[1,0]), */
  /* # coder.typeof(false,[inf,1],[1,0])} */
  /* 'eigenanalysis_surf:23' nv = int32(size(As,3)); */
  nv = As->size[2];

  /* 'eigenanalysis_surf:25' if nargin<4 */
  /* 'eigenanalysis_surf:25' us = coder.nullcopy(zeros(nv,3)); */
  ix = us->size[0] * us->size[1];
  us->size[0] = nv;
  us->size[1] = 3;
  emxEnsureCapacity((emxArray__common *)us, ix, (int32_T)sizeof(real_T));

  /* 'eigenanalysis_surf:27' if nargout>1 */
  /* 'eigenanalysis_surf:27' Vs = coder.nullcopy(zeros(3,3,nv)); */
  ix = Vs->size[0] * Vs->size[1] * Vs->size[2];
  Vs->size[0] = 3;
  Vs->size[1] = 3;
  Vs->size[2] = nv;
  emxEnsureCapacity((emxArray__common *)Vs, ix, (int32_T)sizeof(real_T));

  /* 'eigenanalysis_surf:28' if nargout>2 */
  /* 'eigenanalysis_surf:28' tranks = coder.nullcopy(zeros(nv,1,'int8')); */
  ix = tranks->size[0];
  tranks->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)tranks, ix, (int32_T)sizeof(int8_T));

  /*  Tangent ranks */
  /* 'eigenanalysis_surf:29' if nargout>3 */
  /* 'eigenanalysis_surf:31' tol = 0.003; */
  /*  Loop through vertices one by one to compute displacements */
  /* 'eigenanalysis_surf:34' for jj=1:nv */
  for (jj = 0; jj + 1 <= nv; jj++) {
    /* 'eigenanalysis_surf:35' if nargin==5 && ~to_update(jj) */
    /* 'eigenanalysis_surf:37' b = bs(1:3,jj); */
    /* 'eigenanalysis_surf:38' if nargout==1 && all(b==0) */
    /*  Perform eigenvalue decomposition and sort eigenvalues and eigenvectors */
    /* 'eigenanalysis_surf:41' A = As(1:3,1:3,jj); */
    /* 'eigenanalysis_surf:42' [V,ls] = eig3_sorted(A); */
    for (ix = 0; ix < 3; ix++) {
      for (iy = 0; iy < 3; iy++) {
        b_As[iy + 3 * ix] = As->data[(iy + As->size[0] * ix) + As->size[0] *
          As->size[1] * jj];
      }
    }

    eig3_sorted(b_As, V, ls);

    /* 'eigenanalysis_surf:43' if ls(1)==0 */
    if (ls[0] == 0.0) {
    } else {
      /* 'eigenanalysis_surf:45' sign = V(1:3,1)'*b; */
      for (ix = 0; ix < 3; ix++) {
        a[ix] = V[ix];
      }

      b_sign = 0.0;
      ix = 0;
      iy = 0;
      for (k = 0; k < 3; k++) {
        b_sign += a[ix] * bs->data[iy + bs->size[0] * jj];
        ix++;
        iy++;
      }

      /* 'eigenanalysis_surf:46' d = sign/ls(1)*V(1:3,1); */
      y = b_sign / ls[0];
      for (ix = 0; ix < 3; ix++) {
        d[ix] = y * V[ix];
      }

      /* 'eigenanalysis_surf:48' trank = int8(2); */
      trank = 2;

      /* 'eigenanalysis_surf:50' if nargin>2 && size(isridge,1)>=nv && isridge(jj) && ls(2)>ls(1)*1.e-8 || ... */
      /* 'eigenanalysis_surf:51'             ls(2)>ls(1)*tol || abs(sign)<0.7*sqrt(b'*b) */
      guard1 = FALSE;
      if (((isridge->size[0] >= nv) && isridge->data[jj] && (ls[1] > ls[0] *
            1.0E-8)) || (ls[1] > ls[0] * 0.003)) {
        guard1 = TRUE;
      } else {
        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += bs->data[ix + bs->size[0] * jj] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        if (fabs(b_sign) < 0.7 * sqrt(y)) {
          guard1 = TRUE;
        }
      }

      if (guard1 == TRUE) {
        /*  Ridge vertex */
        /* 'eigenanalysis_surf:52' d = d + (V(1:3,2)'*b)/ls(2)*V(1:3,2); */
        for (ix = 0; ix < 3; ix++) {
          a[ix] = V[3 + ix];
        }

        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += a[ix] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        y /= ls[1];
        for (ix = 0; ix < 3; ix++) {
          d[ix] += y * V[3 + ix];
        }

        /* 'eigenanalysis_surf:53' trank = int8(1); */
        trank = 1;
      }

      /* 'eigenanalysis_surf:56' if  ls(3)>ls(1)*tol */
      if (ls[2] > ls[0] * 0.003) {
        /*  Corner */
        /* 'eigenanalysis_surf:57' d = d + (V(1:3,3)'*b)/ls(3)*V(1:3,3); */
        for (ix = 0; ix < 3; ix++) {
          a[ix] = V[6 + ix];
        }

        y = 0.0;
        ix = 0;
        iy = 0;
        for (k = 0; k < 3; k++) {
          y += a[ix] * bs->data[iy + bs->size[0] * jj];
          ix++;
          iy++;
        }

        y /= ls[2];
        for (ix = 0; ix < 3; ix++) {
          d[ix] += y * V[6 + ix];
        }

        /* 'eigenanalysis_surf:58' trank = int8(0); */
        trank = 0;
      }

      /* 'eigenanalysis_surf:61' us(jj,1:3) = d'; */
      for (ix = 0; ix < 3; ix++) {
        us->data[jj + us->size[0] * ix] = d[ix];
      }

      /* 'eigenanalysis_surf:63' if nargout>1 */
      /* 'eigenanalysis_surf:63' Vs(1:3,1:3,jj)=V; */
      for (ix = 0; ix < 3; ix++) {
        for (iy = 0; iy < 3; iy++) {
          Vs->data[(iy + Vs->size[0] * ix) + Vs->size[0] * Vs->size[1] * jj] =
            V[iy + 3 * ix];
        }
      }

      /* 'eigenanalysis_surf:64' if nargout>2 */
      /* 'eigenanalysis_surf:64' tranks(jj) = trank; */
      tranks->data[jj] = trank;

      /* 'eigenanalysis_surf:65' if nargout>3 */
    }
  }
}

/*
 *
 */
static boolean_T eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv0[4] = { 'w', 'a', 'l', 'f' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(3 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv0[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
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
 * function [bs, degree] = eval_vander_bivar( us, bs, degree, ws, interp0, guardosc)
 */
static void eval_vander_bivar(const emxArray_real_T *us, emxArray_real_T *bs,
  int32_T *degree, const emxArray_real_T *ws)
{
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T ii;
  int32_T jj;
  emxArray_real_T *b_V;
  int32_T c_V;
  int32_T i11;
  int32_T loop_ub;
  int32_T b_loop_ub;
  emxArray_real_T *ws1;
  real_T A;
  int32_T b_bs[2];
  int32_T c_bs[2];
  emxArray_real_T d_bs;
  emxArray_real_T e_bs;
  int32_T f_bs[2];
  int32_T g_bs[2];
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
  /* 'eval_vander_bivar:10' degree = int32(degree); */
  /* 'eval_vander_bivar:11' assert( isa( degree, 'int32')); */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar:14' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar:15' if nargin<5 */
  /* 'eval_vander_bivar:16' if nargin<6 */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar:19' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
  ncols = (*degree + 2) * (*degree + 1) / 2 - 1;

  /* 'eval_vander_bivar:20' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (*degree > 1)) {
    /* 'eval_vander_bivar:21' degree=degree-1; */
    (*degree)--;

    /* 'eval_vander_bivar:22' ncols = idivide((degree+2)*(degree+1),int32(2))-int32(interp0); */
    ncols = (*degree + 2) * (*degree + 1) / 2 - 1;
  }

  emxInit_real_T(&V, 2);

  /* % Construct matrix */
  /* 'eval_vander_bivar:26' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, *degree, V);

  /* 'eval_vander_bivar:27' if interp0 */
  /* 'eval_vander_bivar:27' V=V(:,2:end); */
  ii = V->size[1];
  if (2 > ii) {
    jj = 0;
    ii = 0;
  } else {
    jj = 1;
  }

  emxInit_real_T(&b_V, 2);
  c_V = V->size[0];
  i11 = b_V->size[0] * b_V->size[1];
  b_V->size[0] = c_V;
  b_V->size[1] = ii - jj;
  emxEnsureCapacity((emxArray__common *)b_V, i11, (int32_T)sizeof(real_T));
  loop_ub = (ii - jj) - 1;
  for (ii = 0; ii <= loop_ub; ii++) {
    b_loop_ub = c_V - 1;
    for (i11 = 0; i11 <= b_loop_ub; i11++) {
      b_V->data[i11 + b_V->size[0] * ii] = V->data[i11 + V->size[0] * (jj + ii)];
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
  /* 'eval_vander_bivar:30' if nargin>3 && ~isempty(ws) */
  b_emxInit_real_T(&ws1, 1);
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar:31' if degree>2 */
    if (*degree > 2) {
      /*  Scale weights to be inversely proportional to distance */
      /* 'eval_vander_bivar:32' ws1 = us(:,1).*us(:,1)+us(:,2).*us(:,2); */
      ii = ws1->size[0];
      ws1->size[0] = us->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));
      loop_ub = us->size[0] - 1;
      for (ii = 0; ii <= loop_ub; ii++) {
        ws1->data[ii] = us->data[ii] * us->data[ii] + us->data[ii + us->size[0]]
          * us->data[ii + us->size[0]];
      }

      /* 'eval_vander_bivar:33' ws1 = ws1 + sum(ws1)/double(npnts)*1.e-2; */
      A = sum(ws1);
      A = A / (real_T)npnts * 0.01;
      ii = ws1->size[0];
      emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));
      loop_ub = ws1->size[0] - 1;
      for (ii = 0; ii <= loop_ub; ii++) {
        ws1->data[ii] += A;
      }

      /* 'eval_vander_bivar:34' if degree<4 */
      if (*degree < 4) {
        /* 'eval_vander_bivar:35' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:36' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:37' ws1(ii) = ws(ii) / sqrt(ws1(ii)); */
            ws1->data[ii] = ws->data[ii] / sqrt(ws1->data[ii]);
          } else {
            /* 'eval_vander_bivar:38' else */
            /* 'eval_vander_bivar:39' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      } else {
        /* 'eval_vander_bivar:42' else */
        /* 'eval_vander_bivar:43' for ii=1:npnts */
        for (ii = 0; ii + 1 <= npnts; ii++) {
          /* 'eval_vander_bivar:44' if ws1(ii)~=0 */
          if (ws1->data[ii] != 0.0) {
            /* 'eval_vander_bivar:45' ws1(ii) = ws(ii) / ws1(ii); */
            ws1->data[ii] = ws->data[ii] / ws1->data[ii];
          } else {
            /* 'eval_vander_bivar:46' else */
            /* 'eval_vander_bivar:47' ws1(ii) = ws(ii); */
            ws1->data[ii] = ws->data[ii];
          }
        }
      }

      /* 'eval_vander_bivar:51' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:52' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:52' V(ii,jj) = V(ii,jj) * ws1(ii); */
          V->data[ii + V->size[0] * jj] *= ws1->data[ii];
        }

        /* 'eval_vander_bivar:53' for jj=1:size(bs,2) */
        /* 'eval_vander_bivar:53' bs(ii,jj) = bs(ii,jj) * ws1(ii); */
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
        d_bs.data[ii] = e_bs.data[ii] * ws1->data[ii];
      }
    } else {
      /* 'eval_vander_bivar:55' else */
      /* 'eval_vander_bivar:56' for ii=1:npnts */
      for (ii = 0; ii + 1 <= npnts; ii++) {
        /* 'eval_vander_bivar:57' for jj=1:ncols */
        for (jj = 0; jj + 1 <= ncols; jj++) {
          /* 'eval_vander_bivar:57' V(ii,jj) = V(ii,jj) * ws(ii); */
          V->data[ii + V->size[0] * jj] *= ws->data[ii];
        }

        /* 'eval_vander_bivar:58' for jj=1:int32(size(bs,2)) */
        /* 'eval_vander_bivar:58' bs(ii,jj) = bs(ii,jj) * ws(ii); */
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
        d_bs.data[ii] = e_bs.data[ii] * ws->data[ii];
      }
    }
  }

  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar:65' ts = coder.nullcopy(zeros(ncols,1)); */
  ii = ws1->size[0];
  ws1->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ws1, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:66' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ws1);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar:69' D = coder.nullcopy(zeros(ncols,1)); */
  ii = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, ii, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar:70' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  ii = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar:73' ncols_sub = ncols; */
  /* 'eval_vander_bivar:74' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (ii < ncols) {
      /* 'eval_vander_bivar:75' degree = degree-1; */
      (*degree)--;

      /* 'eval_vander_bivar:77' if degree==0 */
      if (*degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar:79' bs(:) = 0; */
        ii = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, ii, (int32_T)sizeof(real_T));
        loop_ub = bs->size[0] - 1;
        for (ii = 0; ii <= loop_ub; ii++) {
          bs->data[ii] = 0.0;
        }

        exitg1 = 1U;
      } else {
        /* 'eval_vander_bivar:81' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((*degree + 2) * (*degree + 1)) >> 1U) - 1;
      }
    } else {
      /* % Compute Q'bs */
      /* 'eval_vander_bivar:85' bs = compute_qtb( V, bs, ncols_sub); */
      compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar:88' for i=1:ncols_sub */
      for (ii = 0; ii + 1 <= ncols; ii++) {
        /* 'eval_vander_bivar:88' V(i,i) = D(i); */
        V->data[ii + V->size[0] * ii] = D->data[ii];
      }

      /* 'eval_vander_bivar:89' if guardosc */
      /* 'eval_vander_bivar:91' else */
      /* 'eval_vander_bivar:92' bs = backsolve(V, bs, ncols_sub, ts); */
      backsolve(V, bs, ncols, ws1);
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ws1);
  emxFree_real_T(&V);
}

/*
 * function [bs, deg_out,deg_pnt,deg_qr] = eval_vander_bivar_cmf( us, bs, degree, ws, interp0, safeguard)
 */
static int32_T eval_vander_bivar_cmf(const emxArray_real_T *us, emxArray_real_T *
  bs, int32_T degree, const emxArray_real_T *ws)
{
  int32_T deg_out;
  int32_T npnts;
  int32_T ncols;
  emxArray_real_T *V;
  int32_T ii;
  int32_T loop_ub;
  uint32_T jj;
  int32_T b_bs[2];
  int32_T c_bs[2];
  emxArray_real_T d_bs;
  emxArray_real_T e_bs;
  emxArray_real_T *ts;
  emxArray_real_T *D;
  int32_T exitg1;

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
  /* 'eval_vander_bivar_cmf:14' coder.extrinsic('fprintf') */
  /* 'eval_vander_bivar_cmf:16' assert(isa(degree,'int32')); */
  /* 'eval_vander_bivar_cmf:17' if nargin>4 */
  /* 'eval_vander_bivar_cmf:18' if nargin>5 */
  /*  Determine degree of fitting */
  /* 'eval_vander_bivar_cmf:21' npnts = int32(size(us,1)); */
  npnts = us->size[0];

  /* 'eval_vander_bivar_cmf:22' interp0 = (nargin>4 && interp0); */
  /* 'eval_vander_bivar_cmf:23' if nargin<6 */
  /* 'eval_vander_bivar_cmf:23' safeguard=false; */
  /*  Declaring the degree of output */
  /* 'eval_vander_bivar_cmf:26' deg_out = coder.nullcopy(zeros(1,size(bs,2),'int32')); */
  /*  Determine degree of polynomial */
  /* 'eval_vander_bivar_cmf:30' ncols = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
  ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);

  /* 'eval_vander_bivar_cmf:31' while npnts<ncols && degree>1 */
  while ((npnts < ncols) && (degree > 1)) {
    /* 'eval_vander_bivar_cmf:32' degree=degree-1; */
    degree--;

    /* 'eval_vander_bivar_cmf:33' ncols = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
    ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
  }

  emxInit_real_T(&V, 2);

  /* 'eval_vander_bivar_cmf:35' deg_pnt= degree; */
  /* % Construct matrix */
  /* 'eval_vander_bivar_cmf:37' V = gen_vander_bivar(us, degree); */
  gen_vander_bivar(us, degree, V);

  /* 'eval_vander_bivar_cmf:38' if interp0 */
  /* % Scale rows to assign different weights to different points */
  /* 'eval_vander_bivar_cmf:41' if nargin>3 && ~isempty(ws) */
  if (!(ws->size[0] == 0)) {
    /* 'eval_vander_bivar_cmf:42' for ii=1:npnts */
    for (ii = 0; ii + 1 <= npnts; ii++) {
      /* 'eval_vander_bivar_cmf:43' for jj=1:size(V,2) */
      loop_ub = V->size[1];
      for (jj = 1U; (real_T)jj <= (real_T)loop_ub; jj++) {
        /* 'eval_vander_bivar_cmf:44' V(ii,jj) = V(ii,jj) * ws(ii); */
        V->data[ii + V->size[0] * ((int32_T)jj - 1)] *= ws->data[ii];
      }

      /* 'eval_vander_bivar_cmf:46' for jj=1:size(bs,2) */
      /* 'eval_vander_bivar_cmf:47' bs(ii,jj) = bs(ii,jj) .* ws(ii); */
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

  b_emxInit_real_T(&ts, 1);
  b_emxInit_real_T(&D, 1);

  /* % Scale columns to reduce condition number */
  /* 'eval_vander_bivar_cmf:53' ts = coder.nullcopy(zeros(ncols,1)); */
  npnts = ts->size[0];
  ts->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)ts, npnts, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar_cmf:54' [V, ts] = rescale_matrix(V, ncols, ts); */
  rescale_matrix(V, ncols, ts);

  /* % Perform Householder QR factorization */
  /* 'eval_vander_bivar_cmf:57' D = coder.nullcopy(zeros(ncols,1)); */
  npnts = D->size[0];
  D->size[0] = ncols;
  emxEnsureCapacity((emxArray__common *)D, npnts, (int32_T)sizeof(real_T));

  /* 'eval_vander_bivar_cmf:58' [V, D, rnk] = qr_safeguarded(V, ncols, D); */
  npnts = qr_safeguarded(V, ncols, D);

  /* % Adjust degree of fitting */
  /* 'eval_vander_bivar_cmf:61' ncols_sub = ncols; */
  /* 'eval_vander_bivar_cmf:62' while rnk < ncols_sub */
  do {
    exitg1 = 0U;
    if (npnts < ncols) {
      /* 'eval_vander_bivar_cmf:63' degree = degree-1; */
      degree--;

      /* 'eval_vander_bivar_cmf:64' if degree==0 */
      if (degree == 0) {
        /*  Matrix is singular. Consider surface as flat. */
        /* 'eval_vander_bivar_cmf:66' bs(:) = 0; */
        npnts = bs->size[0];
        emxEnsureCapacity((emxArray__common *)bs, npnts, (int32_T)sizeof(real_T));
        loop_ub = bs->size[0] - 1;
        for (npnts = 0; npnts <= loop_ub; npnts++) {
          bs->data[npnts] = 0.0;
        }

        exitg1 = 1U;
      } else {
        /* 'eval_vander_bivar_cmf:68' ncols_sub = int32(bitshift(uint32((degree+2)*(degree+1)),-1))-int32(interp0); */
        ncols = (int32_T)((uint32_T)((degree + 2) * (degree + 1)) >> 1U);
      }
    } else {
      /* 'eval_vander_bivar_cmf:70' deg_qr = degree; */
      /* % Compute Q'bs */
      /* 'eval_vander_bivar_cmf:72' bs = compute_qtb( V, bs, ncols_sub); */
      compute_qtb(V, bs, ncols);

      /* % Perform backward substitution and scale the solutions. */
      /* 'eval_vander_bivar_cmf:75' for i=1:ncols_sub */
      for (npnts = 0; npnts + 1 <= ncols; npnts++) {
        /* 'eval_vander_bivar_cmf:75' V(i,i) = D(i); */
        V->data[npnts + V->size[0] * npnts] = D->data[npnts];
      }

      /* 'eval_vander_bivar_cmf:76' if safeguard */
      /* 'eval_vander_bivar_cmf:78' else */
      /* 'eval_vander_bivar_cmf:79' bs = backsolve(V, bs, ncols_sub, ts); */
      backsolve(V, bs, ncols, ts);

      /* 'eval_vander_bivar_cmf:80' deg_out(:)=degree; */
      deg_out = degree;
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  emxFree_real_T(&D);
  emxFree_real_T(&ts);
  emxFree_real_T(&V);
  return deg_out;
}

/*
 *
 */
static boolean_T f_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg1;
  static const char_T cv5[3] = { 'C', 'M', 'F' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg1 = 0U;
    if (k + 1 <= 3) {
      if (a[k] != cv5[k]) {
        exitg1 = 1U;
      } else {
        k++;
      }
    } else {
      b_bool = TRUE;
      exitg1 = 1U;
    }
  } while (exitg1 == 0U);

  return b_bool;
}

/*
 * function [fid, nc, loc, dist, proj] = find_parent_triangle(pnt, heid, ps, nrms, tris, opphes, v2he)
 */
static void find_parent_triangle(const real_T pnt[3], int32_T heid, const
  emxArray_real_T *ps, const emxArray_real_T *nrms, const emxArray_int32_T *tris,
  const emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T *fid,
  real_T nc[2], int8_T *loc, real_T *dist, int32_T *proj)
{
  real_T dist_best;
  int32_T fid_best;
  int32_T i;
  real_T nc_best[2];
  int8_T loc_best;
  int32_T queue[32];
  int32_T b_fid;
  int32_T nverts;
  int32_T fid_in;
  static const int8_T iv16[3] = { 2, 3, 1 };

  int32_T exitg2;
  static const int8_T iv17[3] = { 3, 1, 2 };

  int32_T opp;
  int32_T exitg1;

  /*  Find the triangle onto which a point projects. */
  /*     [fid, nc] = find_parent_triangle( pnt, heid, ps, nrms, tris, opphes) */
  /*  Currently, the function only searchs up to 2-ring of the origin vertex */
  /*  of the given halfedge. It can be improved for better efficiency and */
  /*  robustness. */
  /* 'find_parent_triangle:7' fid = heid2fid(heid); */
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  /* 'heid2fid:3' coder.inline('always'); */
  /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
  /* 'find_parent_triangle:7' lid = heid2leid(heid); */
  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  /* 'heid2leid:3' coder.inline('always'); */
  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
  /* 'find_parent_triangle:8' tol_dist = 1.e-6; */
  /* 'find_parent_triangle:9' proj = int32(0); */
  *proj = 0;

  /* % First, handle the most common case by going through the 1-ring of vertex. */
  /* 'find_parent_triangle:11' [fid, nc, loc, dist] = project_onto_one_ring(pnt, fid, lid, ps, nrms, tris, opphes); */
  *fid = (int32_T)((uint32_T)heid >> 2U);
  project_onto_one_ring(pnt, fid, (int32_T)((uint32_T)heid & 3U) + 1, ps, nrms,
                        tris, opphes, nc, loc, dist);

  /* 'find_parent_triangle:12' if dist<=tol_dist */
  if (*dist <= 1.0E-6) {
  } else {
    /* 'find_parent_triangle:13' dist_best = dist; */
    dist_best = *dist;

    /* 'find_parent_triangle:13' fid_best = fid; */
    fid_best = *fid;

    /* 'find_parent_triangle:13' nc_best = nc; */
    for (i = 0; i < 2; i++) {
      nc_best[i] = nc[i];
    }

    /* 'find_parent_triangle:13' loc_best = loc; */
    loc_best = *loc;

    /* % Check 2 ring neighborhood. This part is slow, but it is rarely called. */
    /* 'find_parent_triangle:16' queue = zeros(32,1,'int32'); */
    /* 'find_parent_triangle:17' fid = heid2fid(heid); */
    /*  HEID2FID   Obtains face ID from half-edge ID. */
    /* 'heid2fid:3' coder.inline('always'); */
    /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
    *fid = (int32_T)((uint32_T)heid >> 2U);

    /* 'find_parent_triangle:17' lid = heid2leid(heid); */
    /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
    /* 'heid2leid:3' coder.inline('always'); */
    /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
    /* 'find_parent_triangle:18' [queue, nverts] = obtain_nring_surf( tris(fid,lid), 1, int32(0), tris, opphes, v2he, queue); */
    memset((void *)&queue[0], 0, sizeof(int32_T) << 5);

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
    /* 'obtain_nring_surf:65' if nargin>=9 */
    /* 'obtain_nring_surf:67' fid = heid2fid(v2he(vid)); */
    /*  HEID2FID   Obtains face ID from half-edge ID. */
    /* 'heid2fid:3' coder.inline('always'); */
    /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
    b_fid = (int32_T)((uint32_T)v2he->data[tris->data[(*fid + tris->size[0] *
      (int32_T)((uint32_T)heid & 3U)) - 1] - 1] >> 2U) - 1;

    /* 'obtain_nring_surf:67' lid = heid2leid(v2he(vid)); */
    /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
    /* 'heid2leid:3' coder.inline('always'); */
    /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
    i = (int32_T)((uint32_T)v2he->data[tris->data[(*fid + tris->size[0] *
      (int32_T)((uint32_T)heid & 3U)) - 1] - 1] & 3U);

    /* 'obtain_nring_surf:68' nverts=int32(0); */
    nverts = 0;

    /* 'obtain_nring_surf:68' nfaces=int32(0); */
    /* 'obtain_nring_surf:68' overflow = false; */
    /* 'obtain_nring_surf:70' if ~fid */
    if (!(b_fid + 1 != 0)) {
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
      /* 'obtain_nring_surf:88' hebuf = coder.nullcopy(zeros(maxnv,1, 'int32')); */
      /*  Optimized version for collecting one-ring vertices */
      /* 'obtain_nring_surf:91' if opphes( fid, lid) */
      if (opphes->data[b_fid + opphes->size[0] * i] != 0) {
        /* 'obtain_nring_surf:92' fid_in = fid; */
        fid_in = b_fid + 1;
      } else {
        /* 'obtain_nring_surf:93' else */
        /* 'obtain_nring_surf:94' fid_in = int32(0); */
        fid_in = 0;

        /* 'obtain_nring_surf:96' v = tris(fid, nxt(lid)); */
        /* 'obtain_nring_surf:97' nverts = int32(1); */
        nverts = 1;

        /* 'obtain_nring_surf:97' ngbvs( 1) = v; */
        queue[0] = tris->data[b_fid + tris->size[0] * (iv16[i] - 1)];

        /* 'obtain_nring_surf:99' if ~oneringonly */
      }

      /*  Rotate counterclockwise order around vertex and insert vertices */
      /* 'obtain_nring_surf:103' while 1 */
      do {
        exitg2 = 0U;

        /*  Insert vertx into list */
        /* 'obtain_nring_surf:105' lid_prv = prv(lid); */
        i = iv17[i] - 1;

        /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
        /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
        if (nverts < 32) {
          /* 'obtain_nring_surf:109' nverts = nverts + 1; */
          nverts++;

          /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
          queue[nverts - 1] = tris->data[b_fid + tris->size[0] * i];

          /* 'obtain_nring_surf:111' if ~oneringonly */
        } else {
          /* 'obtain_nring_surf:116' else */
          /* 'obtain_nring_surf:117' overflow = true; */
        }

        /* 'obtain_nring_surf:120' opp = opphes(fid, lid_prv); */
        opp = opphes->data[b_fid + opphes->size[0] * i];

        /* 'obtain_nring_surf:121' fid = heid2fid(opp); */
        /*  HEID2FID   Obtains face ID from half-edge ID. */
        /* 'heid2fid:3' coder.inline('always'); */
        /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
        b_fid = (int32_T)((uint32_T)opphes->data[b_fid + opphes->size[0] * i] >>
                          2U) - 1;

        /* 'obtain_nring_surf:123' if fid == fid_in */
        if (b_fid + 1 == fid_in) {
          exitg2 = 1U;
        } else {
          /* 'obtain_nring_surf:125' else */
          /* 'obtain_nring_surf:126' lid = heid2leid(opp); */
          /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
          /* 'heid2leid:3' coder.inline('always'); */
          /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
          i = (int32_T)((uint32_T)opp & 3U);
        }
      } while (exitg2 == 0U);

      /*  Finished cycle */
      /* 'obtain_nring_surf:130' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
      /* 'obtain_nring_surf:131' if overflow */
    }

    /* 'find_parent_triangle:20' for qid=1:nverts */
    opp = 0;
    do {
      exitg1 = 0U;
      if (opp + 1 <= nverts) {
        /* 'find_parent_triangle:21' fid = heid2fid(v2he(queue(qid))); */
        /*  HEID2FID   Obtains face ID from half-edge ID. */
        /* 'heid2fid:3' coder.inline('always'); */
        /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
        /* 'find_parent_triangle:22' lid = heid2leid(v2he(queue(qid))); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        /* 'find_parent_triangle:24' [fid, nc, loc, dist] = project_onto_one_ring(pnt, fid, lid, ps, nrms, tris, opphes); */
        *fid = (int32_T)((uint32_T)v2he->data[queue[opp] - 1] >> 2U);
        project_onto_one_ring(pnt, fid, (int32_T)((uint32_T)v2he->data[queue[opp]
          - 1] & 3U) + 1, ps, nrms, tris, opphes, nc, loc, dist);

        /* 'find_parent_triangle:25' if dist<=tol_dist */
        if (*dist <= 1.0E-6) {
          exitg1 = 1U;
        } else {
          /* 'find_parent_triangle:27' if dist<dist_best */
          if (*dist < dist_best) {
            /* 'find_parent_triangle:28' dist_best = dist; */
            dist_best = *dist;

            /* 'find_parent_triangle:28' fid_best = fid; */
            fid_best = *fid;

            /* 'find_parent_triangle:28' nc_best = nc; */
            for (i = 0; i < 2; i++) {
              nc_best[i] = nc[i];
            }

            /* 'find_parent_triangle:28' loc_best = loc; */
            loc_best = *loc;
          }

          opp++;
        }
      } else {
        /*  snap to the best triangle */
        /* 'find_parent_triangle:33' if (dist_best > tol_dist ) */
        if (dist_best > 1.0E-6) {
          /* 'find_parent_triangle:34' [nc_best,loc_best,dist_best]= snap_to_trianglebnd(loc_best); */
          /*  Snap the projection of the pnt= p + d to the triangle "fid" */
          /*  fid: Id of the triangle */
          /*  nc: Natural Coordinates */
          /*  loc: location of the projection of pnt w.r.t triangle "fid" */
          /*  dist: shortest distance to either the edges or the vertices to "fid" */
          /* 'snap_to_trianglebnd:7' nc=zeros(2,1); */
          for (i = 0; i < 2; i++) {
            nc[i] = 0.0;
          }

          /* 'snap_to_trianglebnd:8' switch loc */
          switch (loc_best) {
           case 1:
            /* 'snap_to_trianglebnd:9' case 1 */
            /* 'snap_to_trianglebnd:10' nc(1) = 1/2; */
            nc[0] = 0.5;

            /* 'snap_to_trianglebnd:11' nc(2) = 0; */
            nc[1] = 0.0;
            break;

           case 2:
            /* 'snap_to_trianglebnd:12' case 2 */
            /* 'snap_to_trianglebnd:13' nc(1) = 1/2; */
            nc[0] = 0.5;

            /* 'snap_to_trianglebnd:14' nc(2) = 1/2; */
            nc[1] = 0.5;
            break;

           case 3:
            /* 'snap_to_trianglebnd:15' case 3 */
            /* 'snap_to_trianglebnd:16' nc(1) = 0; */
            nc[0] = 0.0;

            /* 'snap_to_trianglebnd:17' nc(2) = 1/2; */
            nc[1] = 0.5;
            break;

           case 4:
            /* 'snap_to_trianglebnd:18' case 4 */
            /* 'snap_to_trianglebnd:19' nc(1) = 0; */
            nc[0] = 0.0;

            /* 'snap_to_trianglebnd:20' nc(2) = 0; */
            nc[1] = 0.0;
            break;

           case 5:
            /* 'snap_to_trianglebnd:21' case 5 */
            /* 'snap_to_trianglebnd:22' nc(1) = 1; */
            nc[0] = 1.0;

            /* 'snap_to_trianglebnd:23' nc(2) = 0; */
            nc[1] = 0.0;
            break;

           case 6:
            /* 'snap_to_trianglebnd:24' case 6 */
            /* 'snap_to_trianglebnd:25' nc(1) = 0; */
            nc[0] = 0.0;

            /* 'snap_to_trianglebnd:26' nc(2) = 1; */
            nc[1] = 1.0;
            break;
          }

          /* 'snap_to_trianglebnd:28' dist = 0; */
          /* 'find_parent_triangle:35' dist=dist_best; */
          *dist = 0.0;

          /* 'find_parent_triangle:35' fid = fid_best; */
          *fid = fid_best;

          /* 'find_parent_triangle:35' nc = nc_best; */
          /* 'find_parent_triangle:35' loc=loc_best; */
          *loc = loc_best;
        } else {
          /* 'find_parent_triangle:39' coder.extrinsic('warning'); */
          /* 'find_parent_triangle:40' if dist_best < 1 */
          if (dist_best < 1.0) {
            /* 'find_parent_triangle:41' warning('Could not find projection. Shortest distance was %g', dist_best); */
            /* 'find_parent_triangle:42' dist=dist_best; */
            *dist = 1.0E-6;

            /* 'find_parent_triangle:42' fid = fid_best; */
            *fid = fid_best;

            /* 'find_parent_triangle:42' nc = nc_best; */
            for (i = 0; i < 2; i++) {
              nc[i] = nc_best[i];
            }

            /* 'find_parent_triangle:42' loc=loc_best; */
            *loc = loc_best;

            /* 'find_parent_triangle:42' proj = int32(1); */
            *proj = 1;
          } else {
            /* 'find_parent_triangle:43' else */
            /* 'find_parent_triangle:44' proj = int32(1); */
            *proj = 1;

            /* 'find_parent_triangle:45' error('Could not find projection. Shortest distance was %g', dist_best); */
          }
        }

        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }
}

/*
 *
 */
static boolean_T g_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv6[7] = { 'c', 'm', 'f', '_', 'n', 'r', 'm' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(6 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv6[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
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
  int32_T i2;
  int32_T c;
  emxArray_real_T *v1;
  emxArray_real_T *c_us;
  emxArray_real_T *v2;
  int32_T p;
  emxArray_real_T *r3;
  emxArray_real_T *y;
  emxArray_real_T *a;
  emxArray_real_T *b_v2;
  int32_T nx;
  int32_T kk2;
  int32_T sz[2];
  static const int8_T iv2[10] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55 };

  emxArray_int32_T *r4;
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
    i2 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = ncols;
    emxEnsureCapacity((emxArray__common *)V, i2, (int32_T)sizeof(real_T));

    /*  Preallocate storage */
    /*  Use tensor product */
    /* 'gen_vander_bivar:63' v1 = gen_vander_univar(us(:,1), degree, [], dderiv); */
    i2 = b_us->size[0];
    b_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)b_us, i2, (int32_T)sizeof(real_T));
    c = us->size[0] - 1;
    for (i2 = 0; i2 <= c; i2++) {
      b_us->data[i2] = us->data[i2];
    }

    emxInit_real_T(&v1, 2);
    b_emxInit_real_T(&c_us, 1);
    gen_vander_univar(b_us, degree, v1);

    /* 'gen_vander_bivar:64' v2 = gen_vander_univar(us(:,2), degree, [], dderiv); */
    i2 = c_us->size[0];
    c_us->size[0] = us->size[0];
    emxEnsureCapacity((emxArray__common *)c_us, i2, (int32_T)sizeof(real_T));
    emxFree_real_T(&b_us);
    c = us->size[0] - 1;
    for (i2 = 0; i2 <= c; i2++) {
      c_us->data[i2] = us->data[i2 + us->size[0]];
    }

    emxInit_real_T(&v2, 2);
    gen_vander_univar(c_us, degree, v2);

    /* 'gen_vander_bivar:66' for p=1:npnts */
    p = 0;
    emxFree_real_T(&c_us);
    emxInit_real_T(&r3, 2);
    emxInit_real_T(&y, 2);
    b_emxInit_real_T(&a, 1);
    emxInit_real_T(&b_v2, 2);
    while (p + 1 <= npnts) {
      /* 'gen_vander_bivar:67' V(p,:) = reshape(v1(p,:)'*v2(p,:),1,ncols); */
      i2 = a->size[0];
      a->size[0] = v1->size[1];
      emxEnsureCapacity((emxArray__common *)a, i2, (int32_T)sizeof(real_T));
      c = v1->size[1] - 1;
      for (i2 = 0; i2 <= c; i2++) {
        a->data[i2] = v1->data[p + v1->size[0] * i2];
      }

      i2 = b_v2->size[0] * b_v2->size[1];
      b_v2->size[0] = 1;
      b_v2->size[1] = v2->size[1];
      emxEnsureCapacity((emxArray__common *)b_v2, i2, (int32_T)sizeof(real_T));
      c = v2->size[1] - 1;
      for (i2 = 0; i2 <= c; i2++) {
        b_v2->data[b_v2->size[0] * i2] = v2->data[p + v2->size[0] * i2];
      }

      i2 = y->size[0] * y->size[1];
      y->size[0] = a->size[0];
      y->size[1] = b_v2->size[1];
      emxEnsureCapacity((emxArray__common *)y, i2, (int32_T)sizeof(real_T));
      c = b_v2->size[1] - 1;
      for (i2 = 0; i2 <= c; i2++) {
        nx = a->size[0] - 1;
        for (kk2 = 0; kk2 <= nx; kk2++) {
          y->data[kk2 + y->size[0] * i2] = a->data[kk2] * b_v2->data[b_v2->size
            [0] * i2];
        }
      }

      nx = y->size[0] * y->size[1];
      for (i2 = 0; i2 < 2; i2++) {
        sz[i2] = 0;
      }

      sz[0] = 1;
      sz[1] = ncols;
      i2 = r3->size[0] * r3->size[1];
      r3->size[0] = 1;
      r3->size[1] = sz[1];
      emxEnsureCapacity((emxArray__common *)r3, i2, (int32_T)sizeof(real_T));
      for (c = 0; c + 1 <= nx; c++) {
        r3->data[c] = y->data[c];
      }

      c = r3->size[1] - 1;
      for (i2 = 0; i2 <= c; i2++) {
        V->data[p + V->size[0] * i2] = r3->data[r3->size[0] * i2];
      }

      p++;
    }

    emxFree_real_T(&b_v2);
    emxFree_real_T(&a);
    emxFree_real_T(&y);
    emxFree_real_T(&r3);
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
    i2 = V->size[0] * V->size[1];
    V->size[0] = npnts;
    V->size[1] = (int32_T)iv2[degree];
    emxEnsureCapacity((emxArray__common *)V, i2, (int32_T)sizeof(real_T));
    c = npnts * iv2[degree] - 1;
    for (i2 = 0; i2 <= c; i2++) {
      V->data[i2] = 0.0;
    }

    /*  Preallocate storage */
    /*     %% Compute rows corresponding to function values */
    /* 'gen_vander_bivar:90' V(1:npnts,1) = 1; */
    if (1 > npnts) {
      i2 = 0;
    } else {
      i2 = npnts;
    }

    emxInit_int32_T(&r4, 1);
    kk2 = r4->size[0];
    r4->size[0] = i2;
    emxEnsureCapacity((emxArray__common *)r4, kk2, (int32_T)sizeof(int32_T));
    c = i2 - 1;
    for (i2 = 0; i2 <= c; i2++) {
      r4->data[i2] = 1 + i2;
    }

    c = r4->size[0];
    emxFree_int32_T(&r4);
    c--;
    for (i2 = 0; i2 <= c; i2++) {
      V->data[i2] = 1.0;
    }

    /* 'gen_vander_bivar:91' V(1:npnts,2:3) = us; */
    for (i2 = 0; i2 < 2; i2++) {
      iv3[i2] = (int8_T)(i2 + 1);
    }

    for (i2 = 0; i2 < 2; i2++) {
      c = us->size[0] - 1;
      for (kk2 = 0; kk2 <= c; kk2++) {
        V->data[kk2 + V->size[0] * iv3[i2]] = us->data[kk2 + us->size[0] * i2];
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
  emxArray_int32_T *r5;

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

  emxInit_int32_T(&r5, 1);
  ncols = r5->size[0];
  r5->size[0] = p;
  emxEnsureCapacity((emxArray__common *)r5, ncols, (int32_T)sizeof(int32_T));
  loop_ub = p - 1;
  for (p = 0; p <= loop_ub; p++) {
    r5->data[p] = 1 + p;
  }

  ncols = r5->size[0];
  emxFree_int32_T(&r5);
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
 *
 */
static boolean_T h_eml_strcmp(const char_T a[3])
{
  boolean_T b_bool;
  int32_T k;
  int32_T exitg2;
  int32_T exitg1;
  static const char_T cv7[7] = { 'C', 'M', 'F', '_', 'N', 'R', 'M' };

  b_bool = FALSE;
  k = 0;
  do {
    exitg2 = 0U;
    if (k + 1 <= 2) {
      if ((int8_T)(1 + (int8_T)((int8_T)k << 1)) != (int8_T)(1 + (int8_T)(6 *
            (int8_T)k))) {
        exitg2 = 1U;
      } else {
        k++;
      }
    } else {
      k = 0;
      exitg2 = 2U;
    }
  } while (exitg2 == 0U);

  if (exitg2 == 1U) {
  } else {
    do {
      exitg1 = 0U;
      if (k + 1 <= 3) {
        if (a[k] != cv7[k]) {
          exitg1 = 1U;
        } else {
          k++;
        }
      } else {
        b_bool = TRUE;
        exitg1 = 1U;
      }
    } while (exitg1 == 0U);
  }

  return b_bool;
}

/*
 * function [us_smooth,energy_total,changeratio] = ismooth_trimesh_cleanmesh...
 * (nv_clean, xs, tris, isridge, flabels, refareas2, mu, check_trank)
 */
static void ismooth_trimesh_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_boolean_T *isridge, const
  emxArray_int32_T *flabels, boolean_T check_trank, emxArray_real_T *us_smooth)
{
  emxArray_real_T *elem_energies;
  emxArray_real_T *grads_smooth;
  emxArray_real_T *Hs_smooth;
  b_emxInit_real_T(&elem_energies, 1);
  emxInit_real_T(&grads_smooth, 2);
  c_emxInit_real_T(&Hs_smooth, 3);

  /*  Perform mesh smoothing of a triangulated mesh. */
  /* 'ismooth_trimesh_cleanmesh:8' coder.inline('never') */
  /* 'ismooth_trimesh_cleanmesh:9' if nargin<4 */
  /* 'ismooth_trimesh_cleanmesh:10' if nargin<5 */
  /* 'ismooth_trimesh_cleanmesh:11' if nargin<6 */
  /* 'ismooth_trimesh_cleanmesh:12' if nargin<7 */
  /* 'ismooth_trimesh_cleanmesh:13' if nargin<8 */
  /*  Loop through faces to compute a smoothing term */
  /* 'ismooth_trimesh_cleanmesh:16' [elem_energies, grads_smooth, Hs_smooth] = ... */
  /* 'ismooth_trimesh_cleanmesh:17'     accumulate_isometry_energy_tri(xs, tris, refareas2, mu); */
  accumulate_isometry_energy_tri(xs, tris, elem_energies, grads_smooth,
    Hs_smooth);

  /* 'ismooth_trimesh_cleanmesh:19' assert( numel(elem_energies)>0); */
  /* 'ismooth_trimesh_cleanmesh:20' energy_total = sum(elem_energies); */
  /*  Project grad and hess onto tangent space */
  /* 'ismooth_trimesh_cleanmesh:23' us_smooth = constrained_smooth_surf_cleanmesh(nv_clean, xs, tris, flabels, isridge, grads_smooth, Hs_smooth, check_trank); */
  c_constrained_smooth_surf_clean(nv_clean, xs, tris, flabels, isridge,
    grads_smooth, Hs_smooth, check_trank, us_smooth);

  /*  Compute ratio of changes */
  /* 'ismooth_trimesh_cleanmesh:26' if nargout>2 */
  emxFree_real_T(&Hs_smooth);
  emxFree_real_T(&grads_smooth);
  emxFree_real_T(&elem_energies);
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
 * function allv_ngbvs = obtain_all_nring_surf( ring, nv, elems, opphes, v2he)
 */
static void obtain_all_nring_surf(real_T ring, int32_T nv, const
  emxArray_int32_T *elems, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, emxArray_int32_T *allv_ngbvs)
{
  real_T d2;
  int32_T minpnts;
  int32_T avepnts;
  emxArray_boolean_T *vtags;
  int32_T i7;
  int32_T loop_ub;
  emxArray_boolean_T *ftags;
  int32_T offset;
  int32_T i;
  emxArray_int32_T *r7;
  emxArray_int32_T *r8;
  int32_T ngbvs[128];
  int32_T nverts;
  int32_T i8;
  int32_T i9;
  emxArray_int32_T *b_allv_ngbvs;

  /*  Obtain the nring-neighbors of all the vertices of a surface mesh */
  /*  ALLV_NGBVS = OBTAIN_ALL_NRING_SURF( RING, NV, ELEMS) */
  /*  ALLV_NGBVS = OBTAIN_ALL_NRING_SURF( RING, NV, ELEMS, OPPHES) */
  /*  ALLV_NGBVS = OBTAIN_ALL_NRING_SURF( RING, NV, ELEMS, OPPHES, V2HE) */
  /*  */
  /*  Input arguments */
  /*     RING: the desired size of the ring */
  /*     NV: number of vertices of curve */
  /*     ELEMS: element connectivity */
  /*     OPPHES: Opposite half-edges (optional) */
  /*     V2HE: Mapping from vertex to an incident half-edge (optional) */
  /*  */
  /*  Output arguments */
  /*     allv_ngbvs: List of neighboring vertices for each vertex in a format */
  /*         similar to mixed elements. Note that for each vertex, the vertex  */
  /*         itself is not included in the list. */
  /*  */
  /*  See also obtain_nring_surf */
  /* 'obtain_all_nring_surf:24' if nargin<4 */
  /* 'obtain_all_nring_surf:25' if nargin<5 */
  /* 'obtain_all_nring_surf:27' switch int32(ring*2) */
  d2 = ring * 2.0;
  if ((d2 < 4.503599627370496E+15) && (d2 > -4.503599627370496E+15)) {
    d2 = d2 < 0.0 ? ceil(d2 - 0.5) : floor(d2 + 0.5);
  }

  switch ((int32_T)d2) {
   case 2:
    /* 'obtain_all_nring_surf:28' case 2 */
    /* 'obtain_all_nring_surf:29' minpnts = int32(4); */
    minpnts = 4;

    /* 'obtain_all_nring_surf:29' avepnts = int32(7); */
    avepnts = 7;
    break;

   case 3:
    /* 'obtain_all_nring_surf:30' case 3 */
    /* 'obtain_all_nring_surf:31' minpnts = int32(8); */
    minpnts = 8;

    /* 'obtain_all_nring_surf:31' avepnts = int32(13); */
    avepnts = 13;
    break;

   case 4:
    /* 'obtain_all_nring_surf:32' case 4 */
    /* 'obtain_all_nring_surf:33' minpnts = int32(12); */
    minpnts = 12;

    /* 'obtain_all_nring_surf:33' avepnts = int32(19); */
    avepnts = 19;
    break;

   case 5:
    /* 'obtain_all_nring_surf:34' case 5 */
    /* 'obtain_all_nring_surf:35' minpnts = int32(18); */
    minpnts = 18;

    /* 'obtain_all_nring_surf:35' avepnts = int32(31); */
    avepnts = 31;
    break;

   case 6:
    /* 'obtain_all_nring_surf:36' case 6 */
    /* 'obtain_all_nring_surf:37' minpnts = int32(25); */
    minpnts = 25;

    /* 'obtain_all_nring_surf:37' avepnts = int32(37); */
    avepnts = 37;
    break;

   case 7:
    /* 'obtain_all_nring_surf:38' case 7 */
    /* 'obtain_all_nring_surf:39' minpnts = int32(33); */
    minpnts = 33;

    /* 'obtain_all_nring_surf:39' avepnts = int32(55); */
    avepnts = 55;
    break;

   case 8:
    /* 'obtain_all_nring_surf:40' case 8 */
    /* 'obtain_all_nring_surf:41' minpnts = int32(43); */
    minpnts = 43;

    /* 'obtain_all_nring_surf:41' avepnts = int32(73); */
    avepnts = 73;
    break;

   default:
    /* 'obtain_all_nring_surf:42' otherwise */
    /* 'obtain_all_nring_surf:43' error('Unsupported ring size'); */
    /* 'obtain_all_nring_surf:44' minpnts = int32(0); */
    minpnts = 0;

    /* 'obtain_all_nring_surf:44' avepnts = int32(0); */
    avepnts = 0;
    break;
  }

  emxInit_boolean_T(&vtags, 1);

  /* 'obtain_all_nring_surf:47' ngbvs = coder.nullcopy(zeros(128,1,'int32')); */
  /* 'obtain_all_nring_surf:48' vtags = false(nv, 1); */
  i7 = vtags->size[0];
  vtags->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)vtags, i7, (int32_T)sizeof(boolean_T));
  loop_ub = nv - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    vtags->data[i7] = FALSE;
  }

  emxInit_boolean_T(&ftags, 1);

  /* 'obtain_all_nring_surf:49' ftags = false(size(elems,1), 1); */
  i7 = ftags->size[0];
  ftags->size[0] = elems->size[0];
  emxEnsureCapacity((emxArray__common *)ftags, i7, (int32_T)sizeof(boolean_T));
  loop_ub = elems->size[0] - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    ftags->data[i7] = FALSE;
  }

  /* 'obtain_all_nring_surf:51' allv_ngbvs = coder.nullcopy(zeros( fix(nv*avepnts*1.1),1,'int32')); */
  i7 = allv_ngbvs->size[0];
  d2 = (real_T)(nv * avepnts) * 1.1;
  d2 = d2 < 0.0 ? ceil(d2 - 0.5) : floor(d2 + 0.5);
  allv_ngbvs->size[0] = (int32_T)d2;
  emxEnsureCapacity((emxArray__common *)allv_ngbvs, i7, (int32_T)sizeof(int32_T));

  /* 'obtain_all_nring_surf:53' offset = int32(1); */
  offset = 0;

  /* 'obtain_all_nring_surf:54' for i=1:nv */
  i = 1;
  emxInit_int32_T(&r7, 1);
  emxInit_int32_T(&r8, 1);
  while (i <= nv) {
    /* 'obtain_all_nring_surf:55' [ngbvs, nverts,vtags,ftags] = obtain_nring_quad(i, ring, minpnts, ... */
    /* 'obtain_all_nring_surf:56'         elems, opphes, v2he, ngbvs, vtags, ftags); */
    nverts = obtain_nring_quad(i, ring, minpnts, elems, opphes, v2he, ngbvs,
      vtags, ftags);

    /* 'obtain_all_nring_surf:58' assert( offset +nverts + 1 <= numel(allv_ngbvs)); */
    /* 'obtain_all_nring_surf:59' allv_ngbvs(offset) = nverts; */
    allv_ngbvs->data[offset] = nverts;

    /* 'obtain_all_nring_surf:60' allv_ngbvs(offset+1:offset+nverts) = ngbvs(1:nverts); */
    if (1 > nverts) {
      i7 = 0;
    } else {
      i7 = nverts;
    }

    i8 = offset + 2;
    i9 = (offset + nverts) + 1;
    if (i8 > i9) {
      i8 = 1;
      i9 = 0;
    }

    avepnts = r7->size[0];
    r7->size[0] = (i9 - i8) + 1;
    emxEnsureCapacity((emxArray__common *)r7, avepnts, (int32_T)sizeof(int32_T));
    loop_ub = i9 - i8;
    for (i9 = 0; i9 <= loop_ub; i9++) {
      r7->data[i9] = i8 + i9;
    }

    i8 = r8->size[0];
    r8->size[0] = i7;
    emxEnsureCapacity((emxArray__common *)r8, i8, (int32_T)sizeof(int32_T));
    loop_ub = i7 - 1;
    for (i7 = 0; i7 <= loop_ub; i7++) {
      r8->data[i7] = 1 + i7;
    }

    loop_ub = r8->size[0] - 1;
    for (i7 = 0; i7 <= loop_ub; i7++) {
      allv_ngbvs->data[r7->data[i7] - 1] = ngbvs[r8->data[i7] - 1];
    }

    /* 'obtain_all_nring_surf:62' offset = offset +nverts + 1; */
    offset = (offset + nverts) + 1;
    i++;
  }

  emxFree_int32_T(&r8);
  emxFree_int32_T(&r7);
  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);

  /* 'obtain_all_nring_surf:65' allv_ngbvs = allv_ngbvs(1:offset-1); */
  if (1 > offset) {
    offset = 0;
  }

  emxInit_int32_T(&b_allv_ngbvs, 1);
  i7 = b_allv_ngbvs->size[0];
  b_allv_ngbvs->size[0] = offset;
  emxEnsureCapacity((emxArray__common *)b_allv_ngbvs, i7, (int32_T)sizeof
                    (int32_T));
  loop_ub = offset - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    b_allv_ngbvs->data[i7] = allv_ngbvs->data[i7];
  }

  i7 = allv_ngbvs->size[0];
  allv_ngbvs->size[0] = b_allv_ngbvs->size[0];
  emxEnsureCapacity((emxArray__common *)allv_ngbvs, i7, (int32_T)sizeof(int32_T));
  loop_ub = b_allv_ngbvs->size[0] - 1;
  for (i7 = 0; i7 <= loop_ub; i7++) {
    allv_ngbvs->data[i7] = b_allv_ngbvs->data[i7];
  }

  emxFree_int32_T(&b_allv_ngbvs);
}

/*
 * function [ngbvs, nverts, vtags, ftags, ngbfs, nfaces] = obtain_nring_quad...
 *     ( vid, ring, minpnts, elems, opphes, v2he, ngbvs, vtags, ftags, ngbfs)
 */
static int32_T obtain_nring_quad(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *elems, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], emxArray_boolean_T *vtags,
  emxArray_boolean_T *ftags)
{
  int32_T nverts;
  int32_T fid;
  int32_T lid;
  int32_T nfaces;
  boolean_T overflow;
  boolean_T b7;
  int32_T fid_in;
  static const int8_T nxt[8] = { 2, 2, 3, 3, 1, 4, 0, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T prv[8] = { 3, 4, 1, 1, 2, 2, 0, 3 };

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
  boolean_T b8;
  boolean_T isfirst;
  int32_T exitg3;
  boolean_T guard2 = FALSE;

  /*  OBTAIN_NRING_QUAD Collect n-ring vertices of a quad or mixed mesh. */
  /*  */
  /*  [ngbvs,nverts,vtags,ftags, ngbfs, nfaces] = obtain_nring_quad(vid,ring, ... */
  /*  minpnts,elems,opphes,v2he,ngbvs,vtags,ftags,ngbfs) */
  /*  collects n-ring vertices of a vertex and saves them into NGBVS, where */
  /*  n is a floating point number with 0.5 increments (1, 1.5, 2, etc.). We */
  /*  define the n-ring verticse as follows: */
  /*   - 0-ring: vertex itself */
  /*   - k-ring vertices: vertices that share an edge with (k-1)-ring vertices */
  /*   - (k+0.5)-ring vertices: k-ring plus vertices that share an element */
  /*            with two vertices of k-ring vertices. */
  /*  The function supports not only a pure quad mesh but also a surface mesh */
  /*    with quads and triangles, as long as it stores the fourth vertex of a */
  /*    as zero in the connectivity. Note that for quad meshes, the k-ring */
  /*    vertices do not necessarily form quadilaterals, but (k+0.5)-rings do. */
  /*  */
  /*  Input arguments */
  /*    vid: vertex ID */
  /*    ring: the desired number of rings (it is a float as it can have halves) */
  /*    minpnts: the minimum number of points desired */
  /*    elems: element connectivity */
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
  /*   3. If NGBVS is not enough to store the whole neighborhood, then */
  /*      only a subset of the neighborhood will be returned in NGBVS. */
  /*      The maximum number of points returned is numel(NGBVS) if NGBVS is */
  /*      given as part of the input, or 128 if not an input arguement. */
  /*  */
  /*  See also OBTAIN_NRING_SURF, OBTAIN_NRING_TRI, OBTAIN_NRING_CURV, OBTAIN_NRING_VOL */
  /* 'obtain_nring_quad:51' coder.extrinsic('warning'); */
  /* 'obtain_nring_quad:53' MAXNPNTS = int32(128); */
  /* 'obtain_nring_quad:54' assert( islogical( vtags) && islogical(ftags)); */
  /* 'obtain_nring_quad:56' assert(ring>=1 && floor(ring*2)==ring*2); */
  /* 'obtain_nring_quad:58' fid = heid2fid(v2he(vid)); */
  /*  HEID2FID   Obtains face ID from half-edge ID. */
  /* 'heid2fid:3' coder.inline('always'); */
  /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
  fid = (int32_T)((uint32_T)v2he->data[vid - 1] >> 2U);

  /* 'obtain_nring_quad:58' lid = heid2leid(v2he(vid)); */
  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
  /* 'heid2leid:3' coder.inline('always'); */
  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
  lid = (int32_T)((uint32_T)v2he->data[vid - 1] & 3U);

  /* 'obtain_nring_quad:59' nverts=int32(0); */
  nverts = 0;

  /* 'obtain_nring_quad:59' nfaces=int32(0); */
  nfaces = 0;

  /* 'obtain_nring_quad:59' overflow = false; */
  overflow = FALSE;

  /* 'obtain_nring_quad:61' if ~fid */
  if (!(fid != 0)) {
  } else {
    /* 'obtain_nring_quad:63' nxt = int32([2 3 1 0; 2 3 4 1]); */
    /* 'obtain_nring_quad:64' prv = int32([3 1 2 0; 4 1 2 3]); */
    /* 'obtain_nring_quad:66' if nargin>=7 && ~isempty(ngbvs) */
    /* 'obtain_nring_quad:67' maxnv = int32(numel(ngbvs)); */
    /* 'obtain_nring_quad:71' if nargin>=10 && ~isempty(ngbfs) */
    /* 'obtain_nring_quad:73' else */
    /* 'obtain_nring_quad:74' maxnf = 2*MAXNPNTS; */
    /* 'obtain_nring_quad:74' ngbfs = coder.nullcopy(zeros(maxnf,1, 'int32')); */
    /* 'obtain_nring_quad:77' oneringonly = ring==1 && minpnts==0; */
    if ((ring == 1.0) && (minpnts == 0)) {
      b7 = TRUE;
    } else {
      b7 = FALSE;
    }

    /* 'obtain_nring_quad:78' hebuf = coder.nullcopy(zeros(maxnv,1, 'int32')); */
    /*  Optimized version for collecting one-ring vertices */
    /* 'obtain_nring_quad:81' if opphes( fid, lid) */
    if (opphes->data[(fid + opphes->size[0] * lid) - 1] != 0) {
      /* 'obtain_nring_quad:82' fid_in = fid; */
      fid_in = fid;
    } else {
      /* 'obtain_nring_quad:83' else */
      /* 'obtain_nring_quad:84' fid_in = int32(0); */
      fid_in = 0;

      /*  If vertex is border edge, insert its incident border vertex. */
      /* 'obtain_nring_quad:87' nedges = 3 + (size(elems,2)==4 && elems(fid,end)~=0); */
      /* 'obtain_nring_quad:88' v = elems(fid, nxt( nedges-2, lid)); */
      /* 'obtain_nring_quad:89' nverts = int32(1); */
      nverts = 1;

      /* 'obtain_nring_quad:89' ngbvs( 1) = v; */
      ngbvs[0] = elems->data[(fid + elems->size[0] * (nxt[lid << 1] - 1)) - 1];

      /* 'obtain_nring_quad:91' if ~oneringonly */
      if (!b7) {
        /* 'obtain_nring_quad:91' hebuf(1) = 0; */
        hebuf[0] = 0;
      }
    }

    /*  Rotate counterclockwise order around vertex and insert vertices */
    /* 'obtain_nring_quad:95' while 1 */
    do {
      exitg4 = 0U;

      /* 'obtain_nring_quad:96' nedges = 3 + (size(elems,2)==4 && elems(fid,end)~=0); */
      /* 'obtain_nring_quad:97' lid_prv = prv(nedges-2, lid); */
      lid = prv[lid << 1] - 1;

      /* 'obtain_nring_quad:98' v = elems(fid, lid_prv); */
      /* 'obtain_nring_quad:100' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_quad:101' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_quad:101' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = elems->data[(fid + elems->size[0] * lid) - 1];

        /* 'obtain_nring_quad:103' if ~oneringonly */
        if (!b7) {
          /*  Save a starting edge for newly inserted vertex to allow early */
          /*  termination of rotation around the vertex later. */
          /* 'obtain_nring_quad:106' hebuf(nverts) = fleids2heid(fid, lid_prv); */
          /*  Encode <fid,leid> pair into a heid. */
          /*  HEID = FLEIDS2HEID(FID, LEID) */
          /*  See also HEID2FID, HEID2LEID */
          /* 'fleids2heid:6' heid = fid*4+leid-1; */
          hebuf[nverts - 1] = (fid << 2) + lid;

          /* 'obtain_nring_quad:107' nfaces = nfaces + 1; */
          nfaces++;

          /* 'obtain_nring_quad:107' ngbfs( nfaces) = fid; */
          ngbfs[nfaces - 1] = fid;
        }
      } else {
        /* 'obtain_nring_quad:109' else */
        /* 'obtain_nring_quad:110' overflow = true; */
        overflow = TRUE;
      }

      /* 'obtain_nring_quad:113' opp = opphes(fid, lid_prv); */
      opp = opphes->data[(fid + opphes->size[0] * lid) - 1];

      /* 'obtain_nring_quad:114' fid = heid2fid(opp); */
      /*  HEID2FID   Obtains face ID from half-edge ID. */
      /* 'heid2fid:3' coder.inline('always'); */
      /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
      fid = (int32_T)((uint32_T)opphes->data[(fid + opphes->size[0] * lid) - 1] >>
                      2U);

      /* 'obtain_nring_quad:116' if fid == fid_in */
      if (fid == fid_in) {
        exitg4 = 1U;
      } else {
        /* 'obtain_nring_quad:118' else */
        /* 'obtain_nring_quad:119' lid = heid2leid(opp); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        lid = (int32_T)((uint32_T)opp & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    /* 'obtain_nring_quad:123' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
    if ((ring == 1.0) && ((nverts >= minpnts) || (nverts >= 128) || (nfaces >=
          256))) {
      /* 'obtain_nring_quad:124' if overflow */
    } else {
      /* 'obtain_nring_quad:130' assert( nargin==9); */
      /* 'obtain_nring_quad:131' vtags(vid) = true; */
      vtags->data[vid - 1] = TRUE;

      /* 'obtain_nring_quad:132' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_quad:132' vtags(ngbvs(i))=true; */
        vtags->data[ngbvs[lid - 1] - 1] = TRUE;
      }

      /* 'obtain_nring_quad:133' for i=1:nfaces */
      for (lid = 1; lid <= nfaces; lid++) {
        /* 'obtain_nring_quad:133' ftags(ngbfs(i))=true; */
        ftags->data[ngbfs[lid - 1] - 1] = TRUE;
      }

      /*  Define buffers and prepare tags for further processing */
      /* 'obtain_nring_quad:137' nverts_pre = int32(0); */
      nverts_pre = 0;

      /* 'obtain_nring_quad:138' nfaces_pre = int32(0); */
      nfaces_pre = 0;

      /*  Second, build full-size ring */
      /* 'obtain_nring_quad:141' ring_full = fix( ring); */
      ring_full = ring;
      b_fix(&ring_full);

      /* 'obtain_nring_quad:142' minpnts = min(minpnts, maxnv); */
      minpnts = minpnts <= 128 ? minpnts : 128;

      /* 'obtain_nring_quad:144' cur_ring=1; */
      cur_ring = 1.0;

      /* 'obtain_nring_quad:145' while true */
      do {
        exitg1 = 0U;

        /* 'obtain_nring_quad:146' if cur_ring>ring_full || (cur_ring==ring_full && ring_full~=ring) */
        guard1 = FALSE;
        if ((cur_ring > ring_full) || ((cur_ring == ring_full) && (ring_full !=
              ring))) {
          /*  Collect halfring */
          /* 'obtain_nring_quad:148' nfaces_last = nfaces; */
          opp = nfaces;

          /* 'obtain_nring_quad:148' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_quad:149' for ii = nfaces_pre+1 : nfaces_last */
          while (nfaces_pre + 1 <= opp) {
            /* 'obtain_nring_quad:150' fid = ngbfs(ii); */
            /* 'obtain_nring_quad:152' nedges = 3 + (size(elems,2)==4 && elems(fid,end)~=0); */
            /* 'obtain_nring_quad:153' if nedges == 3 */
            /*  take opposite vertex in opposite face of triangle */
            /* 'obtain_nring_quad:155' for jj=int32(1):3 */
            lid = 0;
            exitg2 = 0U;
            while ((exitg2 == 0U) && (lid + 1 < 4)) {
              /* 'obtain_nring_quad:156' oppe = opphes( ngbfs(ii), jj); */
              /* 'obtain_nring_quad:157' fid = heid2fid(oppe); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                opphes->size[0] * lid) - 1] >> 2U) - 1;

              /* 'obtain_nring_quad:159' if oppe && ~ftags(fid) */
              if ((opphes->data[(ngbfs[nfaces_pre] + opphes->size[0] * lid) - 1]
                   != 0) && (!ftags->data[fid])) {
                /* 'obtain_nring_quad:160' lid = heid2leid(oppe); */
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                /* 'heid2leid:3' coder.inline('always'); */
                /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);

                /* 'obtain_nring_quad:161' v = elems( fid, prv(1, lid)); */
                /* 'obtain_nring_quad:163' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_quad:164'                             ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[elems->data[fid + elems->size[0] *
                                  (prv[lid << 1] - 1)] - 1]) && (nverts >= 128))
                    || ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_quad:165' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_quad:166' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_quad:166' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_quad:167' ftags(fid) = true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_quad:170' if ~vtags(v) && ~overflow */
                if ((!vtags->data[elems->data[fid + elems->size[0] * (prv[lid <<
                      1] - 1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_quad:171' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_quad:171' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = elems->data[fid + elems->size[0] *
                    (prv[lid << 1] - 1)];

                  /* 'obtain_nring_quad:172' vtags(v) = true; */
                  vtags->data[elems->data[fid + elems->size[0] * (prv[lid << 1]
                    - 1)] - 1] = TRUE;
                }

                exitg2 = 1U;
              } else {
                lid++;
              }
            }

            nfaces_pre++;
          }

          /* 'obtain_nring_quad:195' if nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nfaces==nfaces_last */
          if ((nverts >= minpnts) || (nfaces >= 256) || (nfaces == opp)) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_quad:197' else */
            /*  If needs to expand, then undo the last half ring */
            /* 'obtain_nring_quad:199' for i=nverts_last+1:nverts */
            for (lid = nverts_last; lid + 1 <= nverts; lid++) {
              /* 'obtain_nring_quad:199' vtags(ngbvs(i)) = false; */
              vtags->data[ngbvs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_quad:200' nverts = nverts_last; */
            nverts = nverts_last;

            /* 'obtain_nring_quad:202' for i=nfaces_last+1:nfaces */
            for (lid = opp; lid + 1 <= nfaces; lid++) {
              /* 'obtain_nring_quad:202' ftags(ngbfs(i)) = false; */
              ftags->data[ngbfs[lid] - 1] = FALSE;
            }

            /* 'obtain_nring_quad:203' nfaces = nfaces_last; */
            nfaces = opp;
            guard1 = TRUE;
          }
        } else {
          guard1 = TRUE;
        }

        if (guard1 == TRUE) {
          /*  Collect next full level of ring */
          /* 'obtain_nring_quad:208' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_quad:208' nfaces_pre = nfaces; */
          nfaces_pre = nfaces;

          /* 'obtain_nring_quad:209' for ii=nverts_pre+1 : nverts_last */
          while (nverts_pre + 1 <= nverts_last) {
            /* 'obtain_nring_quad:210' v = ngbvs(ii); */
            /* 'obtain_nring_quad:210' fid = heid2fid(v2he(v)); */
            /*  HEID2FID   Obtains face ID from half-edge ID. */
            /* 'heid2fid:3' coder.inline('always'); */
            /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
            fid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] >> 2U) -
              1;

            /* 'obtain_nring_quad:210' lid = heid2leid(v2he(v)); */
            /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
            /* 'heid2leid:3' coder.inline('always'); */
            /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
            lid = (int32_T)((uint32_T)v2he->data[ngbvs[nverts_pre] - 1] & 3U);

            /*  Allow early termination of the loop if an incident halfedge */
            /*  was recorded and the vertex is not incident on a border halfedge */
            /* 'obtain_nring_quad:214' allow_early_term = hebuf(ii) && opphes(fid,lid); */
            if ((hebuf[nverts_pre] != 0) && (opphes->data[fid + opphes->size[0] *
                 lid] != 0)) {
              b8 = TRUE;
            } else {
              b8 = FALSE;
            }

            /* 'obtain_nring_quad:215' if allow_early_term */
            if (b8) {
              /* 'obtain_nring_quad:216' fid = heid2fid(hebuf(ii)); */
              /*  HEID2FID   Obtains face ID from half-edge ID. */
              /* 'heid2fid:3' coder.inline('always'); */
              /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
              fid = (int32_T)((uint32_T)hebuf[nverts_pre] >> 2U) - 1;

              /* 'obtain_nring_quad:216' lid = heid2leid(hebuf(ii)); */
              /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
              /* 'heid2leid:3' coder.inline('always'); */
              /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
              lid = (int32_T)((uint32_T)hebuf[nverts_pre] & 3U);

              /* 'obtain_nring_quad:217' nedges = 3 + (size(elems,2)==4 && elems(fid,end)~=0); */
            }

            /*  Starting point of counterclockwise rotation */
            /* 'obtain_nring_quad:221' if opphes( fid, lid) */
            if (opphes->data[fid + opphes->size[0] * lid] != 0) {
              /* 'obtain_nring_quad:222' fid_in = fid; */
              fid_in = fid + 1;
            } else {
              /* 'obtain_nring_quad:223' else */
              /* 'obtain_nring_quad:224' fid_in = int32(0); */
              fid_in = 0;
            }

            /* 'obtain_nring_quad:226' v = elems(fid, nxt(nedges-2, lid)); */
            /* 'obtain_nring_quad:227' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs); */
            if (overflow || ((!vtags->data[elems->data[fid + elems->size[0] *
                              (nxt[lid << 1] - 1)] - 1]) && (nverts >= 128))) {
              overflow = TRUE;
            } else {
              overflow = FALSE;
            }

            /* 'obtain_nring_quad:228' if ~overflow && ~vtags(v) */
            if ((!overflow) && (!vtags->data[elems->data[fid + elems->size[0] *
                                (nxt[lid << 1] - 1)] - 1])) {
              /* 'obtain_nring_quad:229' nverts = nverts + 1; */
              nverts++;

              /* 'obtain_nring_quad:229' ngbvs( nverts) = v; */
              ngbvs[nverts - 1] = elems->data[fid + elems->size[0] * (nxt[lid <<
                1] - 1)];

              /* 'obtain_nring_quad:229' vtags(v)=true; */
              vtags->data[elems->data[fid + elems->size[0] * (nxt[lid << 1] - 1)]
                - 1] = TRUE;

              /*  Save starting position for next vertex */
              /* 'obtain_nring_quad:231' hebuf(nverts) = fleids2heid(fid, nxt(nedges-2, lid)); */
              /*  Encode <fid,leid> pair into a heid. */
              /*  HEID = FLEIDS2HEID(FID, LEID) */
              /*  See also HEID2FID, HEID2LEID */
              /* 'fleids2heid:6' heid = fid*4+leid-1; */
              hebuf[nverts - 1] = (((fid + 1) << 2) + nxt[lid << 1]) - 1;
            }

            /*  Rotate counterclockwise around the vertex. */
            /* 'obtain_nring_quad:235' isfirst=true; */
            isfirst = TRUE;

            /* 'obtain_nring_quad:236' while true */
            do {
              exitg3 = 0U;

              /*  Insert vertx into list */
              /* 'obtain_nring_quad:238' nedges = 3 + (size(elems,2)==4 && elems(fid,end)~=0); */
              /* 'obtain_nring_quad:239' lid_prv = prv(nedges-2, lid); */
              lid = prv[lid << 1] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_quad:242' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_quad:243' if allow_early_term && ~isfirst */
                if (b8 && (!isfirst)) {
                  exitg3 = 1U;
                } else {
                  guard2 = TRUE;
                }
              } else {
                /* 'obtain_nring_quad:244' else */
                /*  If the face has already been inserted, then the vertex */
                /*  must be inserted already. */
                /* 'obtain_nring_quad:247' v = elems(fid, lid_prv); */
                /* 'obtain_nring_quad:248' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_quad:249'                     ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!vtags->data[elems->data[fid + elems->size[0] *
                                  lid] - 1]) && (nverts >= 128)) ||
                    ((!ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_quad:251' if ~vtags(v) && ~overflow */
                if ((!vtags->data[elems->data[fid + elems->size[0] * lid] - 1]) &&
                    (!overflow)) {
                  /* 'obtain_nring_quad:252' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_quad:252' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = elems->data[fid + elems->size[0] * lid];

                  /* 'obtain_nring_quad:252' vtags(v)=true; */
                  vtags->data[elems->data[fid + elems->size[0] * lid] - 1] =
                    TRUE;

                  /*  Save starting position for next ring */
                  /* 'obtain_nring_quad:255' hebuf(nverts) = fleids2heid(fid, lid_prv); */
                  /*  Encode <fid,leid> pair into a heid. */
                  /*  HEID = FLEIDS2HEID(FID, LEID) */
                  /*  See also HEID2FID, HEID2LEID */
                  /* 'fleids2heid:6' heid = fid*4+leid-1; */
                  hebuf[nverts - 1] = ((fid + 1) << 2) + lid;
                }

                /* 'obtain_nring_quad:258' if ~ftags(fid) && ~overflow */
                if ((!ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_quad:259' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_quad:259' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_quad:259' ftags(fid)=true; */
                  ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_quad:262' isfirst = false; */
                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                /* 'obtain_nring_quad:265' opp = opphes(fid, lid_prv); */
                opp = opphes->data[fid + opphes->size[0] * lid];

                /* 'obtain_nring_quad:266' fid = heid2fid(opp); */
                /*  HEID2FID   Obtains face ID from half-edge ID. */
                /* 'heid2fid:3' coder.inline('always'); */
                /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid] >> 2U) - 1;

                /* 'obtain_nring_quad:268' if fid == fid_in */
                if (fid + 1 == fid_in) {
                  /*  Finished cycle */
                  exitg3 = 1U;
                } else {
                  /* 'obtain_nring_quad:270' else */
                  /* 'obtain_nring_quad:271' lid = heid2leid(opp); */
                  /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                  /* 'heid2leid:3' coder.inline('always'); */
                  /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                  lid = (int32_T)((uint32_T)opp & 3U);
                }
              }
            } while (exitg3 == 0U);

            nverts_pre++;
          }

          /* 'obtain_nring_quad:276' cur_ring = cur_ring+1; */
          cur_ring++;

          /* 'obtain_nring_quad:277' if (nverts>=minpnts && cur_ring>=ring) || nfaces==nfaces_pre || overflow */
          if (((nverts >= minpnts) && (cur_ring >= ring)) || (nfaces ==
               nfaces_pre) || overflow) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_quad:281' nverts_pre = nverts_last; */
            nverts_pre = nverts_last;
          }
        }
      } while (exitg1 == 0U);

      /*  Reset flags */
      /* 'obtain_nring_quad:285' vtags(vid) = false; */
      vtags->data[vid - 1] = FALSE;

      /* 'obtain_nring_quad:286' for i=1:nverts */
      for (lid = 1; lid <= nverts; lid++) {
        /* 'obtain_nring_quad:286' vtags(ngbvs(i))=false; */
        vtags->data[ngbvs[lid - 1] - 1] = FALSE;
      }

      /* 'obtain_nring_quad:287' if ~oneringonly */
      if (!b7) {
        /* 'obtain_nring_quad:287' for i=1:nfaces */
        for (lid = 1; lid <= nfaces; lid++) {
          /* 'obtain_nring_quad:287' ftags(ngbfs(i))=false; */
          ftags->data[ngbfs[lid - 1] - 1] = FALSE;
        }
      }

      /* 'obtain_nring_quad:289' if overflow */
    }
  }

  return nverts;
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
  boolean_T b3;
  int32_T fid_in;
  static const int8_T iv28[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv29[3] = { 3, 1, 2 };

  int32_T opp;
  int32_T nverts_pre;
  int32_T nfaces_pre;
  real_T ring_full;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T b4;
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
      b3 = TRUE;
    } else {
      b3 = FALSE;
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
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv28[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      if (!b3) {
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
      lid = iv29[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        if (!b3) {
          /*  Save starting position for next vertex */
          /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv29[lid] -
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
                                  (iv29[lid] - 1)] - 1]) && (nverts >= 128)) ||
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
                if ((!vtags->data[tris->data[fid + tris->size[0] * (iv29[lid] -
                      1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv29[lid]
                    - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  vtags->data[tris->data[fid + tris->size[0] * (iv29[lid] - 1)]
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
              b4 = TRUE;
            } else {
              b4 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b4) {
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
                                (iv28[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                nverts++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv28[lid]
                  - 1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                vtags->data[tris->data[fid + tris->size[0] * (iv28[lid] - 1)] -
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
              lid = iv29[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b4 && (!isfirst)) {
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
                    (iv29[lid] - 1)];
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
      if (!b3) {
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
 * function [pnt, deg_out] = polyfit3d_cmf_edge(ngbpnts1, nrms1, ngbpnts2, nrms2, xi, deg)
 */
static void polyfit3d_cmf_edge(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, real_T xi, int32_T deg, real_T pnt[3])
{
  real_T s1;
  int32_T k;
  real_T pos[3];
  real_T nrm[3];
  real_T y;
  int32_T ix;
  int32_T iy;
  real_T b_pnt;
  real_T absnrm[3];
  static const int8_T iv22[3] = { 0, 1, 0 };

  static const int8_T iv23[3] = { 1, 0, 0 };

  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws;
  real_T t2[3];
  int32_T np1;
  int32_T np2;
  int32_T j;
  emxArray_real_T *buf;
  boolean_T tc1;
  emxArray_int32_T *r13;
  emxArray_int32_T *r14;
  boolean_T tc2;
  emxArray_int32_T *r15;
  emxArray_real_T *b_bs;

  /*  Compute the position of a point along an edge using polynomial */
  /*          fitting with continuous moving frames.     */
  /*  */
  /*  Input: */
  /*  ngbpnts1-2:Input points of size mx3, Its first column is x-coordinates, */
  /*             and its second column is y-coordinates. The first vertex will */
  /*             be used as the origin of the local coordinate system. */
  /*  nrms1-2:   The normals at ngbptns */
  /*  xi:        The parameter between 0 and 1 */
  /*              */
  /*  deg:       The degree of polynomial to fit, from 0 to 6 */
  /*  h1,h2:     Optional arguments specifying resolution at each point. */
  /*  */
  /*  Output: */
  /*  pnt:    The interpolated coordinates in the global coordinate system */
  /*  deg_out:   The actual degree used in computing the point */
  /*  */
  /*  See also polyfit3d_cmf_tri, polyfit3d_cmf_quad, polyfit3d_cmf_curv */
  /* 'polyfit3d_cmf_edge:21' MAXNPNTS=128; */
  /* 'polyfit3d_cmf_edge:22' coder.varsize( 'bs', 'ws', [2*MAXNPNTS,1], [true,false]); */
  /* 'polyfit3d_cmf_edge:23' coder.varsize( 'us', [2*MAXNPNTS,2], [true,false]); */
  /*  Use quadratic fitting by default */
  /* 'polyfit3d_cmf_edge:26' if nargin<6|| deg==0 */
  if (deg == 0) {
    /* 'polyfit3d_cmf_edge:26' deg=int32(2); */
    deg = 2;
  }

  /*  Compute the position on the triangle face and weighted average normal */
  /* 'polyfit3d_cmf_edge:29' s1 = (1-xi); */
  s1 = 1.0 - xi;

  /* 'polyfit3d_cmf_edge:29' s2 = xi; */
  /* 'polyfit3d_cmf_edge:30' pos = s1.*ngbpnts1(1,1:3) + s2*ngbpnts2(1,1:3); */
  for (k = 0; k < 3; k++) {
    pos[k] = s1 * ngbpnts1->data[ngbpnts1->size[0] * k] + xi * ngbpnts2->
      data[ngbpnts2->size[0] * k];
  }

  /* 'polyfit3d_cmf_edge:31' nrm = s1.*nrms1(1,1:3) + s2.*nrms2(1,1:3); */
  for (k = 0; k < 3; k++) {
    nrm[k] = s1 * nrms1->data[nrms1->size[0] * k] + xi * nrms2->data[nrms2->
      size[0] * k];
  }

  /* 'polyfit3d_cmf_edge:32' nrm = nrm.'./sqrt(nrm*nrm.' + 1.e-100); */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += nrm[ix] * nrm[iy];
    ix++;
    iy++;
  }

  y = sqrt(y + 1.0E-100);

  /* 'polyfit3d_cmf_edge:34' absnrm = abs(nrm); */
  for (k = 0; k < 3; k++) {
    b_pnt = nrm[k] / y;
    absnrm[k] = fabs(b_pnt);
    pnt[k] = b_pnt;
  }

  /*  construct local coordinate [t1,t2,nrm] */
  /* 'polyfit3d_cmf_edge:36' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
  if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
    /* 'polyfit3d_cmf_edge:37' t1 = double([0; 1; 0]); */
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] = (real_T)iv22[ix];
    }
  } else {
    /* 'polyfit3d_cmf_edge:38' else */
    /* 'polyfit3d_cmf_edge:39' t1 = double([1; 0; 0]); */
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] = (real_T)iv23[ix];
    }
  }

  /* 'polyfit3d_cmf_edge:41' t1 = t1 - t1' * nrm * nrm; */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += absnrm[ix] * pnt[iy];
    ix++;
    iy++;
  }

  for (k = 0; k < 3; k++) {
    absnrm[k] -= y * pnt[k];
  }

  /* 'polyfit3d_cmf_edge:41' t1 = t1 / sqrt(t1'*t1); */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += absnrm[ix] * absnrm[iy];
    ix++;
    iy++;
  }

  y = sqrt(y);
  for (k = 0; k < 3; k++) {
    absnrm[k] /= y;
  }

  emxInit_real_T(&us, 2);
  b_emxInit_real_T(&bs, 1);
  b_emxInit_real_T(&ws, 1);

  /* 'polyfit3d_cmf_edge:42' t2 = cross_col( nrm, t1); */
  /* CROSS_COL Efficient routine for computing cross product of two  */
  /* 3-dimensional column vectors. */
  /*  CROSS_COL(A,B) Efficiently computes the cross product between */
  /*  3-dimensional column vector A, and 3-dimensional column vector B. */
  /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
  t2[0] = pnt[1] * absnrm[2] - pnt[2] * absnrm[1];
  t2[1] = pnt[2] * absnrm[0] - pnt[0] * absnrm[2];
  t2[2] = pnt[0] * absnrm[1] - pnt[1] * absnrm[0];

  /* 'polyfit3d_cmf_edge:44' np1 = int32(size(ngbpnts1,1)); */
  np1 = ngbpnts1->size[0];

  /* 'polyfit3d_cmf_edge:44' np2 = int32(size(ngbpnts2,1)); */
  np2 = ngbpnts2->size[0];

  /* 'polyfit3d_cmf_edge:45' us = coder.nullcopy(zeros(np1+np2,2)); */
  k = us->size[0] * us->size[1];
  us->size[0] = np1 + np2;
  us->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)us, k, (int32_T)sizeof(real_T));

  /* 'polyfit3d_cmf_edge:46' bs = coder.nullcopy(zeros(np1+np2,1)); */
  k = bs->size[0];
  bs->size[0] = np1 + np2;
  emxEnsureCapacity((emxArray__common *)bs, k, (int32_T)sizeof(real_T));

  /* 'polyfit3d_cmf_edge:47' ws = coder.nullcopy(zeros(np1+np2,1)); */
  k = ws->size[0];
  ws->size[0] = np1 + np2;
  emxEnsureCapacity((emxArray__common *)ws, k, (int32_T)sizeof(real_T));

  /* % compute right hand side */
  /* 'polyfit3d_cmf_edge:50' for j = 1:np1 */
  for (j = 0; j + 1 <= np1; j++) {
    /* 'polyfit3d_cmf_edge:51' uu = ngbpnts1(j,:)-pos; */
    for (k = 0; k < 3; k++) {
      nrm[k] = ngbpnts1->data[j + ngbpnts1->size[0] * k] - pos[k];
    }

    /* 'polyfit3d_cmf_edge:52' us(j,1) = uu*t1; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[j] = y;

    /* 'polyfit3d_cmf_edge:52' us(j,2) = uu*t2; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[j + us->size[0]] = y;

    /* 'polyfit3d_cmf_edge:53' bs(j) = uu*nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * pnt[iy];
      ix++;
      iy++;
    }

    bs->data[j] = y;
  }

  /* 'polyfit3d_cmf_edge:56' for j = 1: np2 */
  for (j = 0; j + 1 <= np2; j++) {
    /* 'polyfit3d_cmf_edge:57' uu = ngbpnts2(j,:)-pos; */
    for (k = 0; k < 3; k++) {
      nrm[k] = ngbpnts2->data[j + ngbpnts2->size[0] * k] - pos[k];
    }

    /* 'polyfit3d_cmf_edge:58' us(np1+j,1) = uu*t1; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[np1 + j] = y;

    /* 'polyfit3d_cmf_edge:58' us(np1+j,2) = uu*t2; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[(np1 + j) + us->size[0]] = y;

    /* 'polyfit3d_cmf_edge:59' bs(np1+j) = uu*nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * pnt[iy];
      ix++;
      iy++;
    }

    bs->data[np1 + j] = y;
  }

  b_emxInit_real_T(&buf, 1);

  /* % compute weights */
  /* 'polyfit3d_cmf_edge:63' tol=0.707106781186548; */
  /* if nargin<7; h1 = compute_resolution( ngbpnts1); end */
  /* 'polyfit3d_cmf_edge:65' [buf,tc1] = compute_cmf_weights( pos, ngbpnts1, nrms1, deg, false, tol); */
  tc1 = compute_cmf_weights(pos, ngbpnts1, nrms1, deg, buf);

  /* 'polyfit3d_cmf_edge:66' ws(1:np1) = s1*buf; */
  if (1 > np1) {
    k = 0;
  } else {
    k = np1;
  }

  emxInit_int32_T(&r13, 1);
  iy = r13->size[0];
  r13->size[0] = k;
  emxEnsureCapacity((emxArray__common *)r13, iy, (int32_T)sizeof(int32_T));
  ix = k - 1;
  for (k = 0; k <= ix; k++) {
    r13->data[k] = 1 + k;
  }

  b_emxInit_int32_T(&r14, 2);
  k = r14->size[0] * r14->size[1];
  r14->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r14, k, (int32_T)sizeof(int32_T));
  ix = r13->size[0];
  k = r14->size[0] * r14->size[1];
  r14->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)r14, k, (int32_T)sizeof(int32_T));
  ix = r13->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    r14->data[k] = r13->data[k];
  }

  emxFree_int32_T(&r13);
  ix = r14->size[0] * r14->size[1] - 1;
  for (k = 0; k <= ix; k++) {
    ws->data[r14->data[k] - 1] = s1 * buf->data[k];
  }

  /* if nargin<8; h2 = compute_resolution( ngbpnts2); end */
  /* 'polyfit3d_cmf_edge:69' [buf,tc2] = compute_cmf_weights( pos, ngbpnts2, nrms2, deg, false, tol); */
  tc2 = compute_cmf_weights(pos, ngbpnts2, nrms2, deg, buf);

  /* 'polyfit3d_cmf_edge:70' ws(np1+1: np1+np2) = s2*buf; */
  k = np1 + 1;
  iy = np1 + np2;
  if (k > iy) {
    k = 1;
    iy = 0;
  }

  emxInit_int32_T(&r15, 1);
  ix = r15->size[0];
  r15->size[0] = (iy - k) + 1;
  emxEnsureCapacity((emxArray__common *)r15, ix, (int32_T)sizeof(int32_T));
  ix = iy - k;
  for (iy = 0; iy <= ix; iy++) {
    r15->data[iy] = k + iy;
  }

  k = r14->size[0] * r14->size[1];
  r14->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r14, k, (int32_T)sizeof(int32_T));
  ix = r15->size[0];
  k = r14->size[0] * r14->size[1];
  r14->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)r14, k, (int32_T)sizeof(int32_T));
  ix = r15->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    r14->data[k] = r15->data[k];
  }

  emxFree_int32_T(&r15);
  ix = r14->size[0] * r14->size[1] - 1;
  for (k = 0; k <= ix; k++) {
    ws->data[r14->data[k] - 1] = xi * buf->data[k];
  }

  emxFree_real_T(&buf);
  emxFree_int32_T(&r14);

  /* % Solve linear system */
  /*  Use no more than quadratic fitting for too coarse meshes. */
  /* 'polyfit3d_cmf_edge:74' if tc1||tc2 */
  if ((tc1 || tc2) && (deg > 2)) {
    /* 'polyfit3d_cmf_edge:74' deg = min(deg,2); */
    deg = 2;
  }

  b_emxInit_real_T(&b_bs, 1);

  /* 'polyfit3d_cmf_edge:75' [bs, deg_out] = eval_vander_bivar_cmf(us, bs, deg, ws); */
  k = b_bs->size[0];
  b_bs->size[0] = bs->size[0];
  emxEnsureCapacity((emxArray__common *)b_bs, k, (int32_T)sizeof(real_T));
  ix = bs->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    b_bs->data[k] = bs->data[k];
  }

  emxFree_real_T(&bs);
  eval_vander_bivar_cmf(us, b_bs, deg, ws);

  /* % Change back to global coordinate system. */
  /* 'polyfit3d_cmf_edge:78' pnt = pos.' + bs(1)*nrm; */
  y = b_bs->data[0];
  emxFree_real_T(&b_bs);
  emxFree_real_T(&ws);
  emxFree_real_T(&us);
  for (k = 0; k < 3; k++) {
    pnt[k] = pos[k] + y * pnt[k];
  }
}

/*
 * function [pnt, deg_out] = polyfit3d_cmf_tri(ngbpnts1, nrms1, ngbpnts2, nrms2, ...
 *     ngbpnts3, nrms3, xi, eta, deg)
 */
static void polyfit3d_cmf_tri(const emxArray_real_T *ngbpnts1, const
  emxArray_real_T *nrms1, const emxArray_real_T *ngbpnts2, const emxArray_real_T
  *nrms2, const emxArray_real_T *ngbpnts3, const emxArray_real_T *nrms3, real_T
  xi, real_T eta, int32_T deg, real_T pnt[3])
{
  real_T s1;
  int32_T k;
  real_T pos[3];
  real_T nrm[3];
  real_T y;
  int32_T ix;
  int32_T iy;
  real_T absnrm[3];
  static const int8_T iv18[3] = { 0, 1, 0 };

  static const int8_T iv19[3] = { 1, 0, 0 };

  emxArray_real_T *us;
  emxArray_real_T *bs;
  emxArray_real_T *ws;
  real_T t2[3];
  int32_T np1;
  int32_T np2;
  int32_T np3;
  int32_T j;
  int32_T j2;
  emxArray_real_T *buf;
  boolean_T tc1;
  emxArray_int32_T *r9;
  emxArray_int32_T *r10;
  boolean_T tc2;
  emxArray_int32_T *r11;
  boolean_T tc3;
  emxArray_int32_T *r12;
  emxArray_real_T *b_bs;

  /*  Compute the position of a point within a triangle using polynomial  */
  /*          fitting with continuous moving frames. */
  /*  */
  /*  Input: */
  /*  ngbpnts1-3:Input points of size mx3, Its first column is x-coordinates, */
  /*             and its second column is y-coordinates. The first vertex will */
  /*             be used as the origin of the local coordinate system. */
  /*  nrms1-3:   The normals at ngbptns */
  /*  xi,eta:    The two parameters in the tangent plane */
  /*  deg:       The degree of polynomial to fit, from 1 to 6 */
  /*  h1,h2,h3:  Optional arguments specifying resolution at each point. */
  /*  */
  /*  Output: */
  /*  pnt:       The reconstructed point in the global coordinate system */
  /*  deg_out:   The actual degree used in computing the point */
  /*  */
  /*  See also polyfit3d_cmf_quad, polyfit3d_cmf_edge */
  /* 'polyfit3d_cmf_tri:21' MAXNPNTS = 128; */
  /* 'polyfit3d_cmf_tri:22' coder.varsize( 'bs', 'ws', [3*MAXNPNTS,1], [true,false]); */
  /* 'polyfit3d_cmf_tri:23' coder.varsize( 'us', [3*MAXNPNTS,2], [true,false]); */
  /*  Use quadratic fitting by default */
  /* 'polyfit3d_cmf_tri:26' if nargin<9 || deg==0 */
  if (deg == 0) {
    /* 'polyfit3d_cmf_tri:26' deg = int32(2); */
    deg = 2;
  }

  /*  Compute the position on the triangle face and weighted average normal */
  /* 'polyfit3d_cmf_tri:29' s1 = (1-xi-eta); */
  s1 = (1.0 - xi) - eta;

  /* 'polyfit3d_cmf_tri:29' s2 = xi; */
  /* 'polyfit3d_cmf_tri:29' s3 = eta; */
  /* 'polyfit3d_cmf_tri:30' pos = s1.*ngbpnts1(1,1:3) + s2*ngbpnts2(1,1:3) + s3*ngbpnts3(1,1:3); */
  for (k = 0; k < 3; k++) {
    pos[k] = (s1 * ngbpnts1->data[ngbpnts1->size[0] * k] + xi * ngbpnts2->
              data[ngbpnts2->size[0] * k]) + eta * ngbpnts3->data[ngbpnts3->
      size[0] * k];
  }

  /* 'polyfit3d_cmf_tri:31' nrm = s1.*nrms1(1,1:3) + s2.*nrms2(1,1:3) + s3.*nrms3(1,1:3); */
  for (k = 0; k < 3; k++) {
    nrm[k] = (s1 * nrms1->data[nrms1->size[0] * k] + xi * nrms2->data
              [nrms2->size[0] * k]) + eta * nrms3->data[nrms3->size[0] * k];
  }

  /* 'polyfit3d_cmf_tri:32' nrm = nrm.'./sqrt(nrm*nrm.'+1.e-100); */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += nrm[ix] * nrm[iy];
    ix++;
    iy++;
  }

  y = sqrt(y + 1.0E-100);
  for (k = 0; k < 3; k++) {
    pnt[k] = nrm[k] / y;
  }

  /* 'polyfit3d_cmf_tri:34' absnrm = abs(nrm); */
  b_abs(pnt, absnrm);

  /*  construct local coordinate [t1,t2,nrm] */
  /* 'polyfit3d_cmf_tri:36' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
  if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
    /* 'polyfit3d_cmf_tri:37' t1 = [0; 1; 0]; */
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] = (real_T)iv18[ix];
    }
  } else {
    /* 'polyfit3d_cmf_tri:38' else */
    /* 'polyfit3d_cmf_tri:39' t1 = [1; 0; 0]; */
    for (ix = 0; ix < 3; ix++) {
      absnrm[ix] = (real_T)iv19[ix];
    }
  }

  /* 'polyfit3d_cmf_tri:41' t1 = t1 - t1' * nrm * nrm; */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += absnrm[ix] * pnt[iy];
    ix++;
    iy++;
  }

  for (k = 0; k < 3; k++) {
    absnrm[k] -= y * pnt[k];
  }

  /* 'polyfit3d_cmf_tri:41' t1 = t1 / sqrt(t1'*t1); */
  y = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    y += absnrm[ix] * absnrm[iy];
    ix++;
    iy++;
  }

  y = sqrt(y);
  for (k = 0; k < 3; k++) {
    absnrm[k] /= y;
  }

  emxInit_real_T(&us, 2);
  b_emxInit_real_T(&bs, 1);
  b_emxInit_real_T(&ws, 1);

  /* 'polyfit3d_cmf_tri:42' t2 = cross_col( nrm, t1); */
  /* CROSS_COL Efficient routine for computing cross product of two  */
  /* 3-dimensional column vectors. */
  /*  CROSS_COL(A,B) Efficiently computes the cross product between */
  /*  3-dimensional column vector A, and 3-dimensional column vector B. */
  /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
  t2[0] = pnt[1] * absnrm[2] - pnt[2] * absnrm[1];
  t2[1] = pnt[2] * absnrm[0] - pnt[0] * absnrm[2];
  t2[2] = pnt[0] * absnrm[1] - pnt[1] * absnrm[0];

  /* 'polyfit3d_cmf_tri:44' np1 = int32(size(ngbpnts1,1)); */
  np1 = ngbpnts1->size[0];

  /* 'polyfit3d_cmf_tri:44' np2 = int32(size(ngbpnts2,1)); */
  np2 = ngbpnts2->size[0];

  /* 'polyfit3d_cmf_tri:44' np3 = int32(size(ngbpnts3,1)); */
  np3 = ngbpnts3->size[0];

  /* 'polyfit3d_cmf_tri:45' us = coder.nullcopy(zeros(np1+np2+np3,2)); */
  k = us->size[0] * us->size[1];
  us->size[0] = (np1 + np2) + np3;
  us->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)us, k, (int32_T)sizeof(real_T));

  /* 'polyfit3d_cmf_tri:46' bs = coder.nullcopy(zeros(np1+np2+np3,1)); */
  k = bs->size[0];
  bs->size[0] = (np1 + np2) + np3;
  emxEnsureCapacity((emxArray__common *)bs, k, (int32_T)sizeof(real_T));

  /* 'polyfit3d_cmf_tri:47' ws = coder.nullcopy(zeros(np1+np2+np3,1)); */
  k = ws->size[0];
  ws->size[0] = (np1 + np2) + np3;
  emxEnsureCapacity((emxArray__common *)ws, k, (int32_T)sizeof(real_T));

  /* % compute right hand side */
  /* 'polyfit3d_cmf_tri:50' for j = 1:np1 */
  for (j = 0; j + 1 <= np1; j++) {
    /* 'polyfit3d_cmf_tri:51' uu = ngbpnts1(j,:)-pos; */
    for (k = 0; k < 3; k++) {
      nrm[k] = ngbpnts1->data[j + ngbpnts1->size[0] * k] - pos[k];
    }

    /* 'polyfit3d_cmf_tri:52' us(j,1) = uu*t1; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[j] = y;

    /* 'polyfit3d_cmf_tri:52' us(j,2) = uu*t2; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[j + us->size[0]] = y;

    /* 'polyfit3d_cmf_tri:53' bs(j) = uu*nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * pnt[iy];
      ix++;
      iy++;
    }

    bs->data[j] = y;
  }

  /* 'polyfit3d_cmf_tri:56' for j = 1: np2 */
  for (j = 0; j + 1 <= np2; j++) {
    /* 'polyfit3d_cmf_tri:57' uu = ngbpnts2(j,:)-pos; */
    for (k = 0; k < 3; k++) {
      nrm[k] = ngbpnts2->data[j + ngbpnts2->size[0] * k] - pos[k];
    }

    /* 'polyfit3d_cmf_tri:58' us(np1+j,1) = uu*t1; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[np1 + j] = y;

    /* 'polyfit3d_cmf_tri:58' us(np1+j,2) = uu*t2; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[(np1 + j) + us->size[0]] = y;

    /* 'polyfit3d_cmf_tri:59' bs(np1+j) = uu*nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * pnt[iy];
      ix++;
      iy++;
    }

    bs->data[np1 + j] = y;
  }

  /* 'polyfit3d_cmf_tri:62' for j = 1:np3 */
  for (j = 0; j + 1 <= np3; j++) {
    /* 'polyfit3d_cmf_tri:63' j2 = np1+np2+j; */
    j2 = (np1 + np2) + j;

    /* 'polyfit3d_cmf_tri:64' uu = ngbpnts3(j,:)-pos; */
    for (k = 0; k < 3; k++) {
      nrm[k] = ngbpnts3->data[j + ngbpnts3->size[0] * k] - pos[k];
    }

    /* 'polyfit3d_cmf_tri:65' us(j2,1) = uu*t1; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * absnrm[iy];
      ix++;
      iy++;
    }

    us->data[j2] = y;

    /* 'polyfit3d_cmf_tri:65' us(j2,2) = uu*t2; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[j2 + us->size[0]] = y;

    /* 'polyfit3d_cmf_tri:66' bs(j2) = uu*nrm; */
    y = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      y += nrm[ix] * pnt[iy];
      ix++;
      iy++;
    }

    bs->data[j2] = y;
  }

  b_emxInit_real_T(&buf, 1);

  /* % compute weights */
  /* 'polyfit3d_cmf_tri:70' tol=0.707106781186548; */
  /* 'polyfit3d_cmf_tri:71' [buf,tc1] = compute_cmf_weights( pos, ngbpnts1, nrms1, deg, false, tol); */
  tc1 = compute_cmf_weights(pos, ngbpnts1, nrms1, deg, buf);

  /* 'polyfit3d_cmf_tri:72' ws(1:np1) = s1*buf; */
  if (1 > np1) {
    k = 0;
  } else {
    k = np1;
  }

  emxInit_int32_T(&r9, 1);
  iy = r9->size[0];
  r9->size[0] = k;
  emxEnsureCapacity((emxArray__common *)r9, iy, (int32_T)sizeof(int32_T));
  ix = k - 1;
  for (k = 0; k <= ix; k++) {
    r9->data[k] = 1 + k;
  }

  b_emxInit_int32_T(&r10, 2);
  k = r10->size[0] * r10->size[1];
  r10->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r9->size[0];
  k = r10->size[0] * r10->size[1];
  r10->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r9->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    r10->data[k] = r9->data[k];
  }

  emxFree_int32_T(&r9);
  ix = r10->size[0] * r10->size[1] - 1;
  for (k = 0; k <= ix; k++) {
    ws->data[r10->data[k] - 1] = s1 * buf->data[k];
  }

  /* 'polyfit3d_cmf_tri:74' [buf,tc2] = compute_cmf_weights( pos, ngbpnts2, nrms2, deg, false, tol); */
  tc2 = compute_cmf_weights(pos, ngbpnts2, nrms2, deg, buf);

  /* 'polyfit3d_cmf_tri:75' ws(np1+1: np1+np2) = s2*buf; */
  k = np1 + 1;
  iy = np1 + np2;
  if (k > iy) {
    k = 1;
    iy = 0;
  }

  emxInit_int32_T(&r11, 1);
  ix = r11->size[0];
  r11->size[0] = (iy - k) + 1;
  emxEnsureCapacity((emxArray__common *)r11, ix, (int32_T)sizeof(int32_T));
  ix = iy - k;
  for (iy = 0; iy <= ix; iy++) {
    r11->data[iy] = k + iy;
  }

  k = r10->size[0] * r10->size[1];
  r10->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r11->size[0];
  k = r10->size[0] * r10->size[1];
  r10->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r11->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    r10->data[k] = r11->data[k];
  }

  emxFree_int32_T(&r11);
  ix = r10->size[0] * r10->size[1] - 1;
  for (k = 0; k <= ix; k++) {
    ws->data[r10->data[k] - 1] = xi * buf->data[k];
  }

  /* 'polyfit3d_cmf_tri:77' [buf,tc3] = compute_cmf_weights( pos, ngbpnts3, nrms3, deg, false, tol); */
  tc3 = compute_cmf_weights(pos, ngbpnts3, nrms3, deg, buf);

  /* 'polyfit3d_cmf_tri:78' ws(np1+np2+1: np1+np2+np3) = s3*buf; */
  k = (np1 + np2) + 1;
  iy = (np1 + np2) + np3;
  if (k > iy) {
    k = 1;
    iy = 0;
  }

  emxInit_int32_T(&r12, 1);
  ix = r12->size[0];
  r12->size[0] = (iy - k) + 1;
  emxEnsureCapacity((emxArray__common *)r12, ix, (int32_T)sizeof(int32_T));
  ix = iy - k;
  for (iy = 0; iy <= ix; iy++) {
    r12->data[iy] = k + iy;
  }

  k = r10->size[0] * r10->size[1];
  r10->size[0] = 1;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r12->size[0];
  k = r10->size[0] * r10->size[1];
  r10->size[1] = ix;
  emxEnsureCapacity((emxArray__common *)r10, k, (int32_T)sizeof(int32_T));
  ix = r12->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    r10->data[k] = r12->data[k];
  }

  emxFree_int32_T(&r12);
  ix = r10->size[0] * r10->size[1] - 1;
  for (k = 0; k <= ix; k++) {
    ws->data[r10->data[k] - 1] = eta * buf->data[k];
  }

  emxFree_real_T(&buf);
  emxFree_int32_T(&r10);

  /* % Solve linear system */
  /* 'polyfit3d_cmf_tri:81' if tc1 || tc2 || tc3 */
  if ((tc1 || tc2 || tc3) && (deg > 2)) {
    /* 'polyfit3d_cmf_tri:81' deg = min(deg,2); */
    deg = 2;
  }

  b_emxInit_real_T(&b_bs, 1);

  /* 'polyfit3d_cmf_tri:82' [bs, deg_out] = eval_vander_bivar_cmf(us, bs, deg, ws); */
  k = b_bs->size[0];
  b_bs->size[0] = bs->size[0];
  emxEnsureCapacity((emxArray__common *)b_bs, k, (int32_T)sizeof(real_T));
  ix = bs->size[0] - 1;
  for (k = 0; k <= ix; k++) {
    b_bs->data[k] = bs->data[k];
  }

  emxFree_real_T(&bs);
  eval_vander_bivar_cmf(us, b_bs, deg, ws);

  /* % Change back to global coordinate system. */
  /* 'polyfit3d_cmf_tri:85' pnt = pos.' + bs(1)*nrm; */
  y = b_bs->data[0];
  emxFree_real_T(&b_bs);
  emxFree_real_T(&ws);
  emxFree_real_T(&us);
  for (k = 0; k < 3; k++) {
    pnt[k] = pos[k] + y * pnt[k];
  }
}

/*
 * function [pnt,deg_out] = polyfit3d_walf_vertex(pnts, nrms, pos, deg, interp)
 */
static void polyfit3d_walf_vertex(const emxArray_real_T *pnts, const
  emxArray_real_T *nrms, const real_T pos[3], int32_T deg, real_T pnt[3])
{
  int32_T iy;
  real_T nrm[3];
  real_T absnrm[3];
  int32_T i;
  static const int8_T iv20[3] = { 0, 1, 0 };

  real_T t1[3];
  static const int8_T iv21[3] = { 1, 0, 0 };

  real_T u;
  int32_T ix;
  int32_T k;
  emxArray_real_T *us;
  emxArray_real_T *bs;
  real_T t2[3];
  int32_T nverts;
  emxArray_real_T *ws_row;
  emxArray_real_T *b_ws_row;
  boolean_T toocoarse;
  emxArray_real_T *us1;
  emxArray_real_T *ws_row1;
  int32_T c_ws_row[2];
  int32_T b_bs[2];
  emxArray_real_T d_ws_row;
  emxArray_real_T c_bs;
  int32_T b_iy;
  int32_T e_ws_row[2];
  emxArray_real_T *d_bs;
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
  /* 'polyfit3d_walf_vertex:23' MAXNPNTS = 128; */
  /* 'polyfit3d_walf_vertex:24' coder.varsize( 'bs', 'bs1','ws_row','ws_row1', [MAXNPNTS,1], [true,false]); */
  /* 'polyfit3d_walf_vertex:25' coder.varsize( 'us', 'us1', [MAXNPNTS,2], [true,false]); */
  /* 'polyfit3d_walf_vertex:27' nrm = nrms(1,1:3).'; */
  for (iy = 0; iy < 3; iy++) {
    nrm[iy] = nrms->data[nrms->size[0] * iy];
  }

  /* 'polyfit3d_walf_vertex:28' absnrm = abs(nrm); */
  b_abs(nrm, absnrm);

  /* 'polyfit3d_walf_vertex:30' if ( absnrm(1)>absnrm(2) && absnrm(1)>absnrm(3)) */
  if ((absnrm[0] > absnrm[1]) && (absnrm[0] > absnrm[2])) {
    /* 'polyfit3d_walf_vertex:31' t1 = double([0; 1; 0]); */
    for (i = 0; i < 3; i++) {
      t1[i] = (real_T)iv20[i];
    }
  } else {
    /* 'polyfit3d_walf_vertex:32' else */
    /* 'polyfit3d_walf_vertex:33' t1 = double([1; 0; 0]); */
    for (i = 0; i < 3; i++) {
      t1[i] = (real_T)iv21[i];
    }
  }

  /* 'polyfit3d_walf_vertex:36' t1 = t1 - t1' * nrm * nrm; */
  u = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    u += t1[ix] * nrm[iy];
    ix++;
    iy++;
  }

  for (iy = 0; iy < 3; iy++) {
    t1[iy] -= u * nrm[iy];
  }

  /* 'polyfit3d_walf_vertex:36' t1 = t1 / sqrt(t1'*t1); */
  u = 0.0;
  ix = 0;
  iy = 0;
  for (k = 0; k < 3; k++) {
    u += t1[ix] * t1[iy];
    ix++;
    iy++;
  }

  u = sqrt(u);
  for (iy = 0; iy < 3; iy++) {
    t1[iy] /= u;
  }

  emxInit_real_T(&us, 2);
  b_emxInit_real_T(&bs, 1);

  /* 'polyfit3d_walf_vertex:37' t2 = cross_col( nrm, t1); */
  /* CROSS_COL Efficient routine for computing cross product of two  */
  /* 3-dimensional column vectors. */
  /*  CROSS_COL(A,B) Efficiently computes the cross product between */
  /*  3-dimensional column vector A, and 3-dimensional column vector B. */
  /* 'cross_col:7' c = [a(2)*b(3)-a(3)*b(2); a(3)*b(1)-a(1)*b(3); a(1)*b(2)-a(2)*b(1)]; */
  t2[0] = nrm[1] * t1[2] - nrm[2] * t1[1];
  t2[1] = nrm[2] * t1[0] - nrm[0] * t1[2];
  t2[2] = nrm[0] * t1[1] - nrm[1] * t1[0];

  /* % Project onto local coordinate system */
  /* 'polyfit3d_walf_vertex:40' nverts = int32( size(pnts,1)); */
  nverts = pnts->size[0];

  /* 'polyfit3d_walf_vertex:41' us = coder.nullcopy(zeros(nverts-int32(interp),2)); */
  iy = us->size[0] * us->size[1];
  us->size[0] = nverts;
  us->size[1] = 2;
  emxEnsureCapacity((emxArray__common *)us, iy, (int32_T)sizeof(real_T));

  /* 'polyfit3d_walf_vertex:42' bs = coder.nullcopy(zeros(nverts-int32(interp),1)); */
  iy = bs->size[0];
  bs->size[0] = nverts;
  emxEnsureCapacity((emxArray__common *)bs, iy, (int32_T)sizeof(real_T));

  /* 'polyfit3d_walf_vertex:44' us(1,:)=0; */
  for (iy = 0; iy < 2; iy++) {
    us->data[us->size[0] * iy] = 0.0;
  }

  /* 'polyfit3d_walf_vertex:45' for ii=1+int32(interp):nverts */
  for (i = 0; i + 1 <= nverts; i++) {
    /* 'polyfit3d_walf_vertex:46' k = ii-int32(interp); */
    /* 'polyfit3d_walf_vertex:47' uu = pnts(ii,1:3)-pnts(1,1:3); */
    for (iy = 0; iy < 3; iy++) {
      absnrm[iy] = pnts->data[i + pnts->size[0] * iy] - pnts->data[pnts->size[0]
        * iy];
    }

    /* 'polyfit3d_walf_vertex:48' us(k,1) = uu*t1; */
    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += absnrm[ix] * t1[iy];
      ix++;
      iy++;
    }

    us->data[i] = u;

    /* 'polyfit3d_walf_vertex:48' us(k,2) = uu*t2; */
    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += absnrm[ix] * t2[iy];
      ix++;
      iy++;
    }

    us->data[i + us->size[0]] = u;

    /* 'polyfit3d_walf_vertex:49' bs(k) = uu*nrm; */
    u = 0.0;
    ix = 0;
    iy = 0;
    for (k = 0; k < 3; k++) {
      u += absnrm[ix] * nrm[iy];
      ix++;
      iy++;
    }

    bs->data[i] = u;
  }

  b_emxInit_real_T(&ws_row, 1);
  b_emxInit_real_T(&b_ws_row, 1);

  /* 'polyfit3d_walf_vertex:52' [ws_row,toocoarse] = compute_weights( us, nrms, deg); */
  toocoarse = compute_weights(us, nrms, deg, b_ws_row);
  iy = ws_row->size[0];
  ws_row->size[0] = b_ws_row->size[0];
  emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
  ix = b_ws_row->size[0] - 1;
  for (iy = 0; iy <= ix; iy++) {
    ws_row->data[iy] = b_ws_row->data[iy];
  }

  /*  Compute the coefficients and store into bs */
  /* 'polyfit3d_walf_vertex:55' if toocoarse */
  if (toocoarse) {
    emxInit_real_T(&us1, 2);
    b_emxInit_real_T(&ws_row1, 1);

    /* 'polyfit3d_walf_vertex:56' us1 = coder.nullcopy(zeros(nverts-int32(interp),2)); */
    iy = us1->size[0] * us1->size[1];
    us1->size[0] = nverts;
    us1->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us1, iy, (int32_T)sizeof(real_T));

    /* 'polyfit3d_walf_vertex:57' bs1 = coder.nullcopy(zeros(nverts-int32(interp),1)); */
    iy = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));

    /* 'polyfit3d_walf_vertex:58' ws_row1 = coder.nullcopy(zeros(nverts-int32(interp),1)); */
    iy = ws_row1->size[0];
    ws_row1->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row1, iy, (int32_T)sizeof(real_T));

    /* 'polyfit3d_walf_vertex:59' index = int32(0); */
    nverts = 0;

    /* 'polyfit3d_walf_vertex:60' for i = 1:int32(size(us,1)) */
    for (i = 0; i + 1 <= us->size[0]; i++) {
      /* 'polyfit3d_walf_vertex:61' if (ws_row(i)>0) */
      if (b_ws_row->data[i] > 0.0) {
        /* 'polyfit3d_walf_vertex:62' index = index + 1; */
        nverts++;

        /* 'polyfit3d_walf_vertex:63' us1(index,:) = us(i,:); */
        for (iy = 0; iy < 2; iy++) {
          us1->data[(nverts + us1->size[0] * iy) - 1] = us->data[i + us->size[0]
            * iy];
        }

        /* 'polyfit3d_walf_vertex:64' bs1(index,:) = bs(i,:); */
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

        /* 'polyfit3d_walf_vertex:65' ws_row1(index)  = ws_row(i); */
        ws_row1->data[nverts - 1] = b_ws_row->data[i];
      }
    }

    /* 'polyfit3d_walf_vertex:68' us = us1(1:index,:); */
    if (1 > nverts) {
      iy = 0;
    } else {
      iy = nverts;
    }

    i = us->size[0] * us->size[1];
    us->size[0] = iy;
    us->size[1] = 2;
    emxEnsureCapacity((emxArray__common *)us, i, (int32_T)sizeof(real_T));
    for (i = 0; i < 2; i++) {
      ix = iy - 1;
      for (b_iy = 0; b_iy <= ix; b_iy++) {
        us->data[b_iy + us->size[0] * i] = us1->data[b_iy + us1->size[0] * i];
      }
    }

    emxFree_real_T(&us1);

    /* 'polyfit3d_walf_vertex:69' bs = bs1(1:index,:); */
    if (1 > nverts) {
      iy = 0;
    } else {
      iy = nverts;
    }

    e_ws_row[0] = ws_row->size[0];
    e_ws_row[1] = 1;
    i = bs->size[0];
    bs->size[0] = iy;
    emxEnsureCapacity((emxArray__common *)bs, i, (int32_T)sizeof(real_T));
    ix = iy - 1;
    for (iy = 0; iy <= ix; iy++) {
      d_ws_row = *ws_row;
      d_ws_row.size = (int32_T *)&e_ws_row;
      d_ws_row.numDimensions = 1;
      bs->data[iy] = d_ws_row.data[iy];
    }

    /* 'polyfit3d_walf_vertex:70' ws_row = ws_row1(1:index); */
    if (1 > nverts) {
      nverts = 0;
    }

    iy = ws_row->size[0];
    ws_row->size[0] = nverts;
    emxEnsureCapacity((emxArray__common *)ws_row, iy, (int32_T)sizeof(real_T));
    ix = nverts - 1;
    for (iy = 0; iy <= ix; iy++) {
      ws_row->data[iy] = ws_row1->data[iy];
    }

    emxFree_real_T(&ws_row1);
  }

  emxFree_real_T(&b_ws_row);
  b_emxInit_real_T(&d_bs, 1);

  /* 'polyfit3d_walf_vertex:73' [bs,deg_out] = eval_vander_bivar_cmf( us, bs, deg, ws_row, interp, true); */
  iy = d_bs->size[0];
  d_bs->size[0] = bs->size[0];
  emxEnsureCapacity((emxArray__common *)d_bs, iy, (int32_T)sizeof(real_T));
  ix = bs->size[0] - 1;
  for (iy = 0; iy <= ix; iy++) {
    d_bs->data[iy] = bs->data[iy];
  }

  emxFree_real_T(&bs);
  nverts = b_eval_vander_bivar_cmf(us, d_bs, deg, ws_row);

  /* % project the point into u-v plane and evaluate its value */
  /* 'polyfit3d_walf_vertex:76' vec = (pos - pnts(1,1:3)).'; */
  emxFree_real_T(&ws_row);
  emxFree_real_T(&us);
  for (iy = 0; iy < 3; iy++) {
    absnrm[iy] = pos[iy] - pnts->data[pnts->size[0] * iy];
  }

  /* 'polyfit3d_walf_vertex:78' u = vec.' * t1; */
  u = 0.0;
  ix = 0;
  iy = 0;

  /* 'polyfit3d_walf_vertex:79' v = vec.' * t2; */
  v = 0.0;
  i = 0;
  b_iy = 0;
  for (k = 0; k < 3; k++) {
    u += absnrm[ix] * t1[iy];
    ix++;
    iy++;
    v += absnrm[i] * t2[b_iy];
    i++;
    b_iy++;
  }

  /*  Evaluate the polynomial */
  /* 'polyfit3d_walf_vertex:82' V = coder.nullcopy(zeros(28,1)); */
  /* 'polyfit3d_walf_vertex:83' V(1) = u; */
  V[0] = u;

  /* 'polyfit3d_walf_vertex:83' V(2) = v; */
  V[1] = v;

  /* 'polyfit3d_walf_vertex:84' jj = int32(2); */
  i = 1;

  /* 'polyfit3d_walf_vertex:86' for kk=2:deg_out */
  for (b_iy = 2; b_iy <= nverts; b_iy++) {
    /* 'polyfit3d_walf_vertex:87' jj = jj + 1; */
    i++;

    /* 'polyfit3d_walf_vertex:87' V(jj) = V(jj-kk)*u; */
    V[i] = V[i - b_iy] * u;

    /* 'polyfit3d_walf_vertex:89' for kk2=1:kk */
    for (ix = 1; ix <= b_iy; ix++) {
      /* 'polyfit3d_walf_vertex:90' jj = jj + 1; */
      i++;

      /* 'polyfit3d_walf_vertex:90' V(jj) = V(jj-kk-1)*v; */
      V[i] = V[(i - b_iy) - 1] * v;
    }
  }

  /* 'polyfit3d_walf_vertex:94' if interp */
  /* 'polyfit3d_walf_vertex:94' else */
  /* 'polyfit3d_walf_vertex:94' height = bs(1); */
  height = d_bs->data[0];

  /* 'polyfit3d_walf_vertex:95' for kk=1:jj */
  for (b_iy = 1; b_iy <= i + 1; b_iy++) {
    /* 'polyfit3d_walf_vertex:96' height = height + bs(kk+1-int32(interp)) * V(kk); */
    height += d_bs->data[b_iy] * V[b_iy - 1];
  }

  emxFree_real_T(&d_bs);

  /* % Change back to global coordinate system. */
  /* 'polyfit3d_walf_vertex:101' pnt = pnts(1,1:3)' + u*t1 + v*t2 + height*nrm; */
  for (iy = 0; iy < 3; iy++) {
    pnt[iy] = ((pnts->data[pnts->size[0] * iy] + u * t1[iy]) + v * t2[iy]) +
      height * nrm[iy];
  }

  /* % Uncomment this part for generating C code of this function */
  /*  np = int32(size(pos,1)); */
  /*  pnt = coder.nullcopy(zeros(np,3)); */
  /*  for i = 1:np */
  /*      vec = (pos(i,:) - pnts(1,:)).'; */
  /*      u = vec.' * t1; */
  /*      v = vec.' * t2;     */
  /*       */
  /*      %Evaluating the polynomial for height function */
  /*      V = coder.nullcopy(zeros(28,1)); */
  /*      V(1) = u; V(2) = v; */
  /*      jj = int32(2); */
  /*      for kk=2:deg_out */
  /*          jj = jj + 1; V(jj) = V(jj-kk)*u; */
  /*          for kk2=1:kk */
  /*              jj = jj + 1; V(jj) = V(jj-kk-1)*v; */
  /*          end */
  /*      end */
  /*      if interp; height = 0; else height = bs(1); end */
  /*      for kk=1:jj */
  /*          height = height + bs(kk+1-int32(interp)) * V(kk);         */
  /*      end         */
  /*      pnt(i,:) = pnts(1,:)' + u*t1 + v*t2 + height*nrm;     */
  /*  end */
}

/*
 * function [nrms,curs,prdirs] = polyfit_lhf_surf_cleanmesh(nv_clean, xs, tris, ...
 * nrms_proj, opphes, v2he, degree, ring, iterfit, interp, nrms, curs, prdirs)
 */
static void polyfit_lhf_surf_cleanmesh(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms_proj, const
  emxArray_int32_T *opphes, const emxArray_int32_T *v2he, int32_T degree, real_T
  ring, emxArray_real_T *nrms, emxArray_real_T *curs, emxArray_real_T *prdirs)
{
  static const int8_T iv26[6] = { 5, 9, 15, 23, 32, 42 };

  int32_T minpnts;
  emxArray_boolean_T *vtags;
  int32_T nv;
  int32_T nverts;
  int32_T minpntsv;
  emxArray_boolean_T *ftags;
  emxArray_int32_T *degs;
  boolean_T b1;
  boolean_T b2;
  int32_T ii;
  real_T ringv;
  int32_T exitg3;
  int32_T ngbfs[256];
  int32_T ngbvs[128];
  int32_T deg;
  real_T nrm[3];
  real_T prcurvs[2];
  real_T maxprdir[3];
  static const int8_T iv27[6] = { 5, 9, 15, 23, 32, 42 };

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
    minpnts = (int32_T)iv26[degree - 1];
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

  emxInit_int32_T(&degs, 1);

  /* 'polyfit_lhf_surf_cleanmesh:60' degs = coder.nullcopy(zeros(nv,1, 'int32')); */
  nverts = degs->size[0];
  degs->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)degs, nverts, (int32_T)sizeof(int32_T));

  /* 'polyfit_lhf_surf_cleanmesh:62' noprdir = isempty(prdirs); */
  b1 = (prdirs->size[0] == 0);

  /* 'polyfit_lhf_surf_cleanmesh:63' normalonly = nargout==1 || degree==1 || (nargin>9 && isempty(curs)) || ... */
  /* 'polyfit_lhf_surf_cleanmesh:64'     (nargin>10 && isempty(curs) && isempty(prdirs)); */
  if ((degree == 1) || (curs->size[0] == 0) || ((curs->size[0] == 0) &&
       (prdirs->size[0] == 0))) {
    b2 = TRUE;
  } else {
    b2 = FALSE;
  }

  /* 'polyfit_lhf_surf_cleanmesh:66' for ii=1:nv_clean */
  for (ii = 1; ii <= nv_clean; ii++) {
    /* 'polyfit_lhf_surf_cleanmesh:67' ringv = ring; */
    ringv = ring;

    /* 'polyfit_lhf_surf_cleanmesh:69' while (1) */
    do {
      exitg3 = 0U;

      /*  Collect neighbor vertices */
      /* 'polyfit_lhf_surf_cleanmesh:71' maxnf = 2*MAXNPNTS; */
      /* 'polyfit_lhf_surf_cleanmesh:71' ngbfs = coder.nullcopy(zeros(maxnf,1, 'int32')); */
      /* 'polyfit_lhf_surf_cleanmesh:72' [ngbvs, nverts, vtags, ftags] = obtain_nring_surf( ii, ringv, minpnts, ... */
      /* 'polyfit_lhf_surf_cleanmesh:73'             tris, opphes, v2he, ngbvs, vtags, ftags,ngbfs); */
      nverts = c_obtain_nring_surf(ii, ringv, minpnts, tris, opphes, v2he, ngbvs,
        vtags, ftags, ngbfs);

      /* 'polyfit_lhf_surf_cleanmesh:75' if normalonly */
      if (b2) {
        /* 'polyfit_lhf_surf_cleanmesh:76' [nrm, deg] = polyfit_lhf_surf_point( ii, ngbvs, nverts, xs, ... */
        /* 'polyfit_lhf_surf_cleanmesh:77'                 nrms_proj, degree, interp); */
        polyfit_lhf_surf_point(ii, ngbvs, nverts, xs, nrms_proj, degree, nrm,
          &deg);
      } else {
        /* 'polyfit_lhf_surf_cleanmesh:78' else */
        /* 'polyfit_lhf_surf_cleanmesh:79' if noprdir */
        if (b1) {
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
        exitg3 = 1U;
      }
    } while (exitg3 == 0U);

    /* 'polyfit_lhf_surf_cleanmesh:97' else */
  }

  /* 'polyfit_lhf_surf_cleanmesh:103' if nargout==1 || (~iterfit && (nargout==2 && ~size(curs,1) || ... */
  /* 'polyfit_lhf_surf_cleanmesh:104'         (nargout==3 && ~size(curs,1) && ~size(prdirs,1)))) */
  if ((!(curs->size[0] != 0)) && (!(prdirs->size[0] != 0))) {
  } else {
    /* 'polyfit_lhf_surf_cleanmesh:108' assert(~isempty(degs)); */
    /* % */
    /* 'polyfit_lhf_surf_cleanmesh:110' if nargout==2 || isempty(prdirs) */
    if (prdirs->size[0] == 0) {
      /* 'polyfit_lhf_surf_cleanmesh:111' if iterfit */
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
              minpntsv = (int32_T)iv27[degree - 1];
            }

            /* 'polyfit_lhfgrad_surf_cleanmesh:52' while (1) */
            do {
              exitg2 = 0U;

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
                  curs->data[(ii + curs->size[0] * nverts) - 1] = prcurvs[nverts];
                }
              }

              /*  Enlarge the neighborhood if necessary */
              /* 'polyfit_lhfgrad_surf_cleanmesh:68' if deg < deg_in && ringv<ring+ring */
              if ((deg < nv) && (ringv < ring + ring)) {
                /* 'polyfit_lhfgrad_surf_cleanmesh:69' ringv=ringv+0.5; */
                ringv += 0.5;
              } else {
                exitg2 = 1U;
              }
            } while (exitg2 == 0U);

            /* 'polyfit_lhfgrad_surf_cleanmesh:70' else */
          }
        }
      }
    } else {
      /* 'polyfit_lhf_surf_cleanmesh:118' else */
      /* 'polyfit_lhf_surf_cleanmesh:119' if iterfit */
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
          b1 = TRUE;
        } else {
          b1 = FALSE;
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
              minpntsv = (int32_T)iv27[degree - 1];
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
              if (b1) {
                /* 'polyfit_lhfgrad_surf_cleanmesh:58' [deg, prcurvs] = polyfit_lhfgrad_surf_point( ii, ngbvs, nverts, xs, nrms, deg_in, false); */
                polyfit_lhfgrad_surf_point(ii, ngbvs, nverts, xs, nrms, nv, &deg,
                  prcurvs);
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
                  curs->data[(ii + curs->size[0] * nverts) - 1] = prcurvs[nverts];
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

  emxFree_int32_T(&degs);
  emxFree_boolean_T(&ftags);
  emxFree_boolean_T(&vtags);
}

/*
 * function [nrm, deg, prcurvs, maxprdir] = polyfit_lhf_surf_point...
 *     ( v, ngbvs, nverts, xs, nrms_coor, degree, interp, guardosc)
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
 * function [fid, nc, loc, dist] = project_onto_one_ring(pnt, fid, lid, ps, nrms, tris, opphes)
 */
static void project_onto_one_ring(const real_T pnt[3], int32_T *fid, int32_T lid,
  const emxArray_real_T *ps, const emxArray_real_T *nrms, const emxArray_int32_T
  *tris, const emxArray_int32_T *opphes, real_T nc[2], int8_T *loc, real_T *dist)
{
  real_T dist_best;
  int32_T fid_best;
  int32_T i;
  real_T nc_best[2];
  int32_T fid_start;
  int8_T loc_best;
  int32_T exitg1;
  real_T d;
  boolean_T exitg2;
  real_T s[3];
  int32_T b_i;
  int32_T ix;
  int32_T iy;
  int32_T i15;
  real_T pn[9];
  real_T J[9];
  real_T b_nrms[9];
  real_T c_nrms[9];
  real_T d_nrms[9];
  real_T T;
  real_T r_neg[3];
  real_T S[3];
  real_T pivot;
  real_T b_S[2];
  boolean_T guard1 = FALSE;
  static const int8_T iv32[3] = { 2, 3, 1 };

  /* 'find_parent_triangle:50' coder.extrinsic('warning'); */
  /* 'find_parent_triangle:52' next = int32([2 3 1]); */
  /* 'find_parent_triangle:53' tol_dist = 1.e-6; */
  /* 'find_parent_triangle:55' dist_best=realmax; */
  dist_best = 1.7976931348623157E+308;

  /* 'find_parent_triangle:55' fid_best=int32(0); */
  fid_best = 0;

  /* 'find_parent_triangle:55' nc_best = [realmax;realmax]; */
  for (i = 0; i < 2; i++) {
    nc_best[i] = 1.7976931348623157E+308;
  }

  /*  Loop through the one-ring around the origin vertex of heid */
  /*  in counterclockwise order, and choose the "best" projection. */
  /* 'find_parent_triangle:59' fid_start = fid; */
  fid_start = *fid;

  /* 'find_parent_triangle:59' loc_best = int8(0); */
  loc_best = 0;

  /* 'find_parent_triangle:60' count = int32(1); */
  /* 'find_parent_triangle:62' while true */
  do {
    exitg1 = 0U;

    /* 'find_parent_triangle:63' pnts_elem = ps(tris(fid,1:3),1:3); */
    /* 'find_parent_triangle:64' nrms_elem = nrms(tris(fid,1:3),1:3); */
    /* 'find_parent_triangle:66' nc = fe2_project_point( pnt, pnts_elem, nrms_elem); */
    /*  Project a given point onto a given triangle or quadrilateral element. */
    /*  */
    /*     [nc, d,inverted] = fe2_project_point( pnt, pnts_elem, nrms_elem, tol) */
    /*  */
    /*  Input arguments */
    /*     pnt: the point to be projected */
    /*     pnts_elem: the points (n-by-3) of the vertices of the element */
    /*     nrms_elem: the normals (n-by-3) at the vertices of the element */
    /*     tol: the stopping criteria for Gauss-Newton iteration for  */
    /*          nonlinear elements. */
    /*  Output arguments */
    /*     nc:  the natural coordinates of the projection of the point */
    /*          within the element */
    /*     inverted: it is true if the prism composed of pnts_elem and  */
    /*          pnts_elem+d*nrms_elems is inverted. It indicates the point is */
    /*          too far from the triangle. */
    /*  */
    /*  The function solves the nonlinear equation */
    /*   pnts_elem'*shapefunc(xi,eta)+d*(nrms_elem'*shapefunc(xi,eta)') = pnt */
    /*  using Newton's method to find xi, eta, and d. */
    /*  */
    /*  See also fe2_natcoor, fe2_shapefunc */
    /* 'fe2_project_point:25' if nargin<4 */
    /* 'fe2_project_point:25' tol=1e-12; */
    /* 'fe2_project_point:27' nvpe = size( pnts_elem,1); */
    /* 'fe2_project_point:28' tol2 = tol*tol; */
    /* 'fe2_project_point:30' d = 0; */
    d = 0.0;

    /* 'fe2_project_point:31' if nvpe==3 */
    /* 'fe2_project_point:32' nc = [0.;0.]; */
    for (i = 0; i < 2; i++) {
      nc[i] = 0.0;
    }

    /* 'fe2_project_point:33' for i=1:5 */
    i = 1;
    exitg2 = 0U;
    while ((exitg2 == 0U) && (i <= 5)) {
      /* 'fe2_project_point:34' [J,N] = Jac(3, nc, d, pnts_elem, nrms_elem); */
      /*  Compute Jacobian matrix with w.r.t. xi, eta, and d. */
      /*  3 columns of J contain partial derivatives w.r.t. xi, eta, and d, respectively */
      /* 'fe2_project_point:76' J = coder.nullcopy(zeros(3,3)); */
      /* 'fe2_project_point:77' if nvpe==3 */
      /* 'fe2_project_point:78' N = [1-nc(1)-nc(2); nc(1); nc(2)]; */
      s[0] = (1.0 - nc[0]) - nc[1];
      s[1] = nc[0];
      s[2] = nc[1];

      /* 'fe2_project_point:79' pn = pnts_elem(1:3,:)+d*nrms_elem(1:3,:); */
      b_i = *fid;
      ix = *fid;
      for (iy = 0; iy < 3; iy++) {
        for (i15 = 0; i15 < 3; i15++) {
          pn[i15 + 3 * iy] = ps->data[(tris->data[(b_i + tris->size[0] * i15) -
            1] + ps->size[0] * iy) - 1] + d * nrms->data[(tris->data[(ix +
            tris->size[0] * i15) - 1] + nrms->size[0] * iy) - 1];
        }
      }

      /* 'fe2_project_point:80' J(:,1) = pn(2,:)-pn(1,:); */
      for (iy = 0; iy < 3; iy++) {
        J[iy] = pn[1 + 3 * iy] - pn[3 * iy];

        /* 'fe2_project_point:81' J(:,2) = pn(3,:)-pn(1,:); */
        J[3 + iy] = pn[2 + 3 * iy] - pn[3 * iy];
      }

      /* 'fe2_project_point:82' J(:,3) = N(1)*nrms_elem(1,:)+N(2)*nrms_elem(2,:)+N(3)*nrms_elem(3,:); */
      b_i = *fid;
      for (iy = 0; iy < 3; iy++) {
        for (i15 = 0; i15 < 3; i15++) {
          b_nrms[i15 + 3 * iy] = nrms->data[(tris->data[(b_i + tris->size[0] *
            i15) - 1] + nrms->size[0] * iy) - 1];
        }
      }

      b_i = *fid;
      for (iy = 0; iy < 3; iy++) {
        for (i15 = 0; i15 < 3; i15++) {
          c_nrms[i15 + 3 * iy] = nrms->data[(tris->data[(b_i + tris->size[0] *
            i15) - 1] + nrms->size[0] * iy) - 1];
        }
      }

      b_i = *fid;
      for (iy = 0; iy < 3; iy++) {
        for (i15 = 0; i15 < 3; i15++) {
          d_nrms[i15 + 3 * iy] = nrms->data[(tris->data[(b_i + tris->size[0] *
            i15) - 1] + nrms->size[0] * iy) - 1];
        }
      }

      for (iy = 0; iy < 3; iy++) {
        J[6 + iy] = (s[0] * b_nrms[3 * iy] + s[1] * c_nrms[1 + 3 * iy]) + s[2] *
          d_nrms[2 + 3 * iy];
      }

      /* 'fe2_project_point:36' r_neg = (pnts_elem' * N + d*J(:,3) - pnt); */
      b_i = *fid;
      for (iy = 0; iy < 3; iy++) {
        for (i15 = 0; i15 < 3; i15++) {
          b_nrms[i15 + 3 * iy] = ps->data[(tris->data[(b_i + tris->size[0] * iy)
            - 1] + ps->size[0] * i15) - 1];
        }
      }

      for (iy = 0; iy < 3; iy++) {
        T = 0.0;
        for (i15 = 0; i15 < 3; i15++) {
          T += b_nrms[iy + 3 * i15] * s[i15];
        }

        r_neg[iy] = (T + d * J[6 + iy]) - pnt[iy];
      }

      /* 'fe2_project_point:37' s = solve3x3(J, r_neg); */
      memcpy((void *)&pn[0], (void *)&J[0], 9U * sizeof(real_T));

      /*  Solves a 3x3 linear system with multiple right-hand side vectors */
      /*  using Gaussian elimination with partial pivoting. */
      /*      xs=solve3x3(A,bs) */
      /*      [xs,det]=solve3x3(A,bs) */
      /*      [xs,det,A,P]=solve3x3(A,bs) */
      /*  A is a 3-by-3 matrix of coefficients, and B is 3-by-k. */
      /*  If xs and bs use the same variable, then bs is passed by reference. */
      /*  If det is specified for output, it also returns the determinant of the matrix. */
      /*  If A is specified for output, then A is passed by reference, and it */
      /*  stores the L and U factors at output, with P storing the permutation vector. */
      /*  */
      /*  See also solve2x2 */
      /* 'solve3x3:15' coder.extrinsic('warning'); */
      /* 'solve3x3:16' if nargout>3 */
      /* 'solve3x3:18' S = abs(A(1:3,1)); */
      for (b_i = 0; b_i < 3; b_i++) {
        s[b_i] = r_neg[b_i];
        S[b_i] = fabs(J[b_i]);
      }

      /* 'solve3x3:19' if S(1)>=S(2) && S(1)>=S(3) */
      if ((S[0] >= S[1]) && (S[0] >= S[2])) {
        /* 'solve3x3:20' pivot = A(1,1); */
        pivot = J[0];

        /* 'solve3x3:21' det = pivot; */
        /* 'solve3x3:22' if (pivot==0) */
      } else if (S[1] >= S[2]) {
        /* 'solve3x3:23' elseif S(2)>=S(3) */
        /* 'solve3x3:24' pivot = A(2,1); */
        pivot = J[1];

        /* 'solve3x3:25' det = -pivot; */
        /* 'solve3x3:26' T = A(2,:); */
        /* 'solve3x3:26' A(2,:)=A(1,:); */
        for (iy = 0; iy < 3; iy++) {
          pn[1 + 3 * iy] = J[3 * iy];

          /* 'solve3x3:26' A(1,:)=T; */
          pn[3 * iy] = J[1 + 3 * iy];
        }

        /* 'solve3x3:27' T = bs(2,:); */
        /* 'solve3x3:27' bs(2,:)=bs(1,:); */
        s[1] = r_neg[0];

        /* 'solve3x3:27' bs(1,:)=T; */
        s[0] = r_neg[1];

        /* 'solve3x3:28' if nargout>3 */
      } else {
        /* 'solve3x3:29' else */
        /* 'solve3x3:30' if nargout>3 */
        /* 'solve3x3:31' pivot = A(3,1); */
        pivot = J[2];

        /* 'solve3x3:32' det = -pivot; */
        /* 'solve3x3:33' T = A(3,:); */
        /* 'solve3x3:33' A(3,:)=A(1,:); */
        for (iy = 0; iy < 3; iy++) {
          pn[2 + 3 * iy] = J[3 * iy];

          /* 'solve3x3:33' A(1,:)=T; */
          pn[3 * iy] = J[2 + 3 * iy];
        }

        /* 'solve3x3:34' T = bs(3,:); */
        /* 'solve3x3:34' bs(3,:)=bs(1,:); */
        s[2] = r_neg[0];

        /* 'solve3x3:34' bs(1,:)=T; */
        s[0] = r_neg[2];

        /* 'solve3x3:35' if nargout>3 */
      }

      /* 'solve3x3:38' A(2,1) = A(2,1)/pivot; */
      pn[1] /= pivot;

      /* 'solve3x3:39' A(2,2:end) = A(2,2:end) - A(2,1)*A(1,2:end); */
      T = pn[1];
      for (iy = 0; iy < 2; iy++) {
        pn[1 + 3 * (1 + iy)] -= T * pn[3 * (1 + iy)];
      }

      /* 'solve3x3:40' bs(2,:) = bs(2,:) - A(2,1)*bs(1,:); */
      s[1] -= pn[1] * s[0];

      /* 'solve3x3:42' A(3,1) = A(3,1)/pivot; */
      pn[2] /= pivot;

      /* 'solve3x3:43' A(3,2:end) = A(3,2:end) - A(3,1)*A(1,2:end); */
      T = pn[2];
      for (iy = 0; iy < 2; iy++) {
        pn[2 + 3 * (1 + iy)] -= T * pn[3 * (1 + iy)];
      }

      /* 'solve3x3:44' bs(3,:) = bs(3,:) - A(3,1)*bs(1,:); */
      s[2] -= pn[2] * s[0];

      /* 'solve3x3:46' S = abs(A(2:3,2)); */
      for (b_i = 0; b_i < 2; b_i++) {
        b_S[b_i] = fabs(pn[b_i + 4]);
      }

      /* 'solve3x3:47' if S(1) >= S(2) */
      if (b_S[0] >= b_S[1]) {
        /* 'solve3x3:48' pivot = A(2,2); */
        pivot = pn[4];

        /* 'solve3x3:49' det = det*pivot; */
        /* 'solve3x3:50' if (pivot==0) */
      } else {
        /* 'solve3x3:51' else */
        /* 'solve3x3:52' pivot = A(3,2); */
        pivot = pn[5];

        /* 'solve3x3:53' det = -det*pivot; */
        /* 'solve3x3:54' T = A(3,:); */
        /* 'solve3x3:54' A(3,:)=A(2,:); */
        for (iy = 0; iy < 3; iy++) {
          T = pn[2 + 3 * iy];
          pn[2 + 3 * iy] = pn[1 + 3 * iy];

          /* 'solve3x3:54' A(2,:)=T; */
          pn[1 + 3 * iy] = T;
        }

        /* 'solve3x3:55' T = bs(3,:); */
        T = s[2];

        /* 'solve3x3:55' bs(3,:)=bs(2,:); */
        s[2] = s[1];

        /* 'solve3x3:55' bs(2,:)=T; */
        s[1] = T;

        /* 'solve3x3:56' if nargout>3 */
      }

      /* 'solve3x3:59' A(3,2) = A(3,2)/pivot; */
      pn[5] /= pivot;

      /* 'solve3x3:60' A(3,3) = A(3,3) - A(3,2)*A(2,3); */
      pn[8] -= pn[5] * pn[7];

      /* 'solve3x3:61' bs(3,:) = bs(3,:) - A(3,2)*bs(2,:); */
      s[2] -= pn[5] * s[1];

      /* 'solve3x3:62' if (A(3,3)== 0) */
      /* 'solve3x3:63' det = det*A(3,3); */
      /* 'solve3x3:65' bs(3,:) = bs(3,:) / A(3,3); */
      s[2] /= pn[8];

      /* 'solve3x3:66' bs(2,:) = (bs(2,:) - A(2,3)*bs(3,:)) / A(2,2); */
      s[1] = (s[1] - pn[7] * s[2]) / pn[4];

      /* 'solve3x3:67' bs(1,:) = (bs(1,:) - A(1,3)*bs(3,:) - A(1,2)*bs(2,:)) / A(1,1); */
      s[0] = ((s[0] - pn[6] * s[2]) - pn[3] * s[1]) / pn[0];

      /* 'fe2_project_point:38' nc = nc-s(1:2); */
      for (iy = 0; iy < 2; iy++) {
        nc[iy] -= s[iy];
      }

      /* 'fe2_project_point:39' d = d-s(3); */
      d -= s[2];

      /* 'fe2_project_point:41' err = s'*s; */
      T = 0.0;
      ix = 0;
      iy = 0;
      for (b_i = 0; b_i < 3; b_i++) {
        T += s[ix] * s[iy];
        ix++;
        iy++;
      }

      /* 'fe2_project_point:42' if err < tol2 */
      if (T < 1.0E-24) {
        exitg2 = 1U;
      } else {
        i++;
      }
    }

    /* 'fe2_project_point:59' if nargout>1 */
    /* 'find_parent_triangle:67' loc = fe2_encode_location( 3, nc, tol_dist ); */
    /*  Encode the location of a point within a triangle or quadrilateral element */
    /*  */
    /*  At input, nc is 1x2 or 2-by-1 and stores the natural coordinates of PNT. */
    /*  The output loc encodes the region as follows: */
    /*  */
    /*                \  6 / */
    /*                 \  /                    8  |    3       |  7 */
    /*                  v3                        |            | */
    /*                  /\                   ----v4------------v3---- */
    /*                 /  \                       |            | */
    /*             3  /    \  2                   |            | */
    /*               /      \                  4  |     0      |  2 */
    /*              /   0    \                    |            | */
    /*             /          \                   |            | */
    /*        ----v1-----------v2------        ---v1----------v2------ */
    /*        4  /              \ 5               |            | */
    /*          /       1        \             5  |     1      |  6 */
    /*  */
    /*  */
    /*  On the boundary of different regions, the higher value takes precedence. */
    /*  */
    /*  See also fe2_shapefunc, fe2_project_point, fe2_benc */
    /* 'fe2_encode_location:25' if nargin<3 */
    /* 'fe2_encode_location:27' if nvpe==3 || nvpe==6 */
    /*  Assign location for triangle */
    /* 'fe2_encode_location:29' nc3 = 1-nc(1)-nc(2); */
    T = (1.0 - nc[0]) - nc[1];

    /* 'fe2_encode_location:30' if nc(1)>tol && nc(2)>tol && nc3>tol */
    if ((nc[0] > 1.0E-6) && (nc[1] > 1.0E-6) && (T > 1.0E-6)) {
      /* 'fe2_encode_location:31' loc = int8(0); */
      *loc = 0;

      /*  Face */
    } else if (nc[0] > 1.0E-6) {
      /* 'fe2_encode_location:32' elseif nc(1)>tol */
      /* 'fe2_encode_location:33' if nc3>tol */
      if (T > 1.0E-6) {
        /* 'fe2_encode_location:34' loc = int8(1); */
        *loc = 1;

        /*  Edge 1 */
      } else if (nc[1] > 1.0E-6) {
        /* 'fe2_encode_location:35' elseif nc(2)>tol */
        /* 'fe2_encode_location:36' loc = int8(2); */
        *loc = 2;

        /*  Edge 2 */
      } else {
        /* 'fe2_encode_location:37' else */
        /* 'fe2_encode_location:38' loc = int8(5); */
        *loc = 5;

        /*  Vertex 2 */
      }
    } else if (nc[1] <= 1.0E-6) {
      /* 'fe2_encode_location:40' elseif nc(2)<=tol */
      /* 'fe2_encode_location:41' loc = int8(4); */
      *loc = 4;

      /*  Vertex 1 */
    } else if (T <= 1.0E-6) {
      /* 'fe2_encode_location:42' elseif nc3<=tol */
      /* 'fe2_encode_location:43' loc = int8(6); */
      *loc = 6;

      /*  Vertex 3 */
    } else {
      /* 'fe2_encode_location:44' else */
      /* 'fe2_encode_location:45' loc = int8(3); */
      *loc = 3;

      /*  Edge 3 */
    }

    /*  compute shortest distance to boundary */
    /* 'find_parent_triangle:70' switch loc */
    guard1 = FALSE;
    switch (*loc) {
     case 0:
      /* 'find_parent_triangle:71' case 0 */
      /* 'find_parent_triangle:72' dist = 0; */
      *dist = 0.0;
      exitg1 = 1U;
      break;

     case 1:
      /* 'find_parent_triangle:73' case 1 */
      /* 'find_parent_triangle:74' dist = -nc(2); */
      *dist = -nc[1];
      guard1 = TRUE;
      break;

     case 2:
      /* 'find_parent_triangle:75' case 2 */
      /* 'find_parent_triangle:76' dist = nc(1)+nc(2)-1; */
      *dist = (nc[0] + nc[1]) - 1.0;
      guard1 = TRUE;
      break;

     case 3:
      /* 'find_parent_triangle:77' case 3 */
      /* 'find_parent_triangle:78' dist = -nc(1); */
      *dist = -nc[0];
      guard1 = TRUE;
      break;

     case 4:
      /* 'find_parent_triangle:79' case 4 */
      /* 'find_parent_triangle:80' dist = sqrt(nc(1)*nc(1)+nc(2)*nc(2)); */
      *dist = sqrt(nc[0] * nc[0] + nc[1] * nc[1]);
      guard1 = TRUE;
      break;

     case 5:
      /* 'find_parent_triangle:81' case 5 */
      /* 'find_parent_triangle:82' dist = sqrt((1-nc(1))*(1-nc(1))+nc(2)*nc(2)); */
      *dist = sqrt((1.0 - nc[0]) * (1.0 - nc[0]) + nc[1] * nc[1]);
      guard1 = TRUE;
      break;

     case 6:
      /* 'find_parent_triangle:83' case 6 */
      /* 'find_parent_triangle:84' dist = sqrt(nc(1)*nc(1)+(1-nc(2))*(1-nc(2))); */
      *dist = sqrt(nc[0] * nc[0] + (1.0 - nc[1]) * (1.0 - nc[1]));
      guard1 = TRUE;
      break;

     default:
      /* 'find_parent_triangle:85' otherwise */
      /* 'find_parent_triangle:86' dist = realmax; */
      *dist = 1.7976931348623157E+308;
      guard1 = TRUE;
      break;
    }

    if (guard1 == TRUE) {
      /* 'find_parent_triangle:88' if dist<tol_dist */
      if (*dist < 1.0E-6) {
        exitg1 = 1U;
      } else {
        /* 'find_parent_triangle:90' if dist<dist_best */
        if (*dist < dist_best) {
          /* 'find_parent_triangle:91' dist_best = dist; */
          dist_best = *dist;

          /* 'find_parent_triangle:91' nc_best = nc; */
          for (i = 0; i < 2; i++) {
            nc_best[i] = nc[i];
          }

          /* 'find_parent_triangle:91' fid_best = fid; */
          fid_best = *fid;

          /* 'find_parent_triangle:91' loc_best = loc; */
          loc_best = *loc;
        }

        /* 'find_parent_triangle:94' fid_next = heid2fid( opphes( fid, lid)); */
        /*  HEID2FID   Obtains face ID from half-edge ID. */
        /* 'heid2fid:3' coder.inline('always'); */
        /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
        b_i = (int32_T)((uint32_T)opphes->data[(*fid + opphes->size[0] * (lid -
          1)) - 1] >> 2U);

        /* 'find_parent_triangle:95' if fid_next ==0 || fid_next == fid_start */
        if ((b_i == 0) || (b_i == fid_start)) {
          /* 'find_parent_triangle:107' fid=fid_best; */
          *fid = fid_best;

          /* 'find_parent_triangle:107' nc=nc_best; */
          for (i = 0; i < 2; i++) {
            nc[i] = nc_best[i];
          }

          /* 'find_parent_triangle:107' loc=loc_best; */
          *loc = loc_best;

          /* 'find_parent_triangle:107' dist=dist_best; */
          *dist = dist_best;
          exitg1 = 1U;
        } else {
          /* 'find_parent_triangle:98' lid_next = next(heid2leid( opphes( fid, lid))); */
          /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
          /* 'heid2leid:3' coder.inline('always'); */
          /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
          lid = (int32_T)iv32[(int32_T)((uint32_T)opphes->data[(*fid +
            opphes->size[0] * (lid - 1)) - 1] & 3U)];

          /* 'find_parent_triangle:100' fid = fid_next; */
          *fid = b_i;

          /* 'find_parent_triangle:100' lid = lid_next; */
          /* 'find_parent_triangle:101' count = count + 1; */
          /* 'find_parent_triangle:102' if (count>100) */
        }
      }
    }
  } while (exitg1 == 0U);
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
 *
 */
static void rdivide(const emxArray_real_T *x, const emxArray_real_T *y,
                    emxArray_real_T *z)
{
  int32_T i5;
  int32_T loop_ub;
  i5 = z->size[0];
  z->size[0] = x->size[0];
  emxEnsureCapacity((emxArray__common *)z, i5, (int32_T)sizeof(real_T));
  loop_ub = x->size[0] - 1;
  for (i5 = 0; i5 <= loop_ub; i5++) {
    z->data[i5] = x->data[i5] / y->data[i5];
  }
}

/*
 *
 */
static void repmat(real_T m, emxArray_real_T *b)
{
  real_T d1;
  int32_T b_m[2];
  int32_T i4;
  int32_T outsize[2];
  int32_T loop_ub;
  d1 = m;
  if ((d1 < 4.503599627370496E+15) && (d1 > -4.503599627370496E+15)) {
    d1 = d1 < 0.0 ? ceil(d1 - 0.5) : floor(d1 + 0.5);
  }

  b_m[0] = (int32_T)d1;
  b_m[1] = 1;
  for (i4 = 0; i4 < 2; i4++) {
    outsize[i4] = b_m[i4];
  }

  i4 = b->size[0];
  b->size[0] = outsize[0];
  emxEnsureCapacity((emxArray__common *)b, i4, (int32_T)sizeof(real_T));
  loop_ub = outsize[0] - 1;
  for (i4 = 0; i4 <= loop_ub; i4++) {
    b->data[i4] = 1.0E-100;
  }
}

/*
 * function [alpha_vs,us] = rescale_displacements(xs, us, tris, tol, alpha_in)
 */
static void rescale_displacements(const emxArray_real_T *xs, const
  emxArray_real_T *us, const emxArray_int32_T *tris, emxArray_real_T *alpha_vs)
{
  int32_T nv;
  int32_T ntri;
  int32_T i17;
  int32_T ii;
  real_T b_us[9];
  boolean_T x[9];
  boolean_T y;
  boolean_T exitg1;
  real_T b_xs[9];
  real_T alpha_tri;
  real_T u0;
  real_T minval[3];

  /* RESCALE_DISPLACEMENTS */
  /*  Given layers of surfaces and the displacement of a particular layer, */
  /*  scale the vertex displacements to avoid folding */
  /* 'async_scale_disps_tri_cleanmesh:40' coder.inline('never') */
  /* 'async_scale_disps_tri_cleanmesh:41' nv   = int32(size(xs,1)); */
  nv = xs->size[0];

  /* 'async_scale_disps_tri_cleanmesh:42' ntri = int32(size(tris,1)); */
  ntri = tris->size[0];

  /* 'async_scale_disps_tri_cleanmesh:43' alpha_vs = ones(nv,1); */
  i17 = alpha_vs->size[0];
  alpha_vs->size[0] = nv;
  emxEnsureCapacity((emxArray__common *)alpha_vs, i17, (int32_T)sizeof(real_T));
  nv--;
  for (i17 = 0; i17 <= nv; i17++) {
    alpha_vs->data[i17] = 1.0;
  }

  /* 'async_scale_disps_tri_cleanmesh:45' for ii=1:ntri */
  for (ii = 0; ii + 1 <= ntri; ii++) {
    /* 'async_scale_disps_tri_cleanmesh:46' vs = tris(ii,1:3); */
    /* 'async_scale_disps_tri_cleanmesh:47' if nargin>6 && all(alpha_in(vs)==1) */
    /* 'async_scale_disps_tri_cleanmesh:52' us_tri = us(vs,1:3); */
    /* 'async_scale_disps_tri_cleanmesh:53' if all(us_tri(:)==0) */
    for (i17 = 0; i17 < 3; i17++) {
      for (nv = 0; nv < 3; nv++) {
        b_us[nv + 3 * i17] = us->data[(tris->data[ii + tris->size[0] * nv] +
          us->size[0] * i17) - 1];
      }
    }

    for (i17 = 0; i17 < 9; i17++) {
      x[i17] = (b_us[i17] == 0.0);
    }

    y = TRUE;
    nv = 1;
    exitg1 = 0U;
    while ((exitg1 == 0U) && (nv <= 9)) {
      if (x[nv - 1] == 0) {
        y = FALSE;
        exitg1 = 1U;
      } else {
        nv++;
      }
    }

    if (y) {
    } else {
      /* 'async_scale_disps_tri_cleanmesh:55' alpha_tri = check_prism( xs(vs,1:3), us_tri); */
      for (i17 = 0; i17 < 3; i17++) {
        for (nv = 0; nv < 3; nv++) {
          b_xs[nv + 3 * i17] = xs->data[(tris->data[ii + tris->size[0] * nv] +
            xs->size[0] * i17) - 1];
        }
      }

      for (i17 = 0; i17 < 3; i17++) {
        for (nv = 0; nv < 3; nv++) {
          b_us[nv + 3 * i17] = us->data[(tris->data[ii + tris->size[0] * nv] +
            us->size[0] * i17) - 1];
        }
      }

      alpha_tri = check_prism(b_xs, b_us);

      /* 'async_scale_disps_tri_cleanmesh:57' if alpha_tri < tol */
      if (alpha_tri < 0.1) {
        /* 'async_scale_disps_tri_cleanmesh:57' alpha_tri = 0.5*alpha_tri; */
        alpha_tri *= 0.5;
      }

      /* 'async_scale_disps_tri_cleanmesh:59' if alpha_tri<1 */
      if (alpha_tri < 1.0) {
        /* 'async_scale_disps_tri_cleanmesh:60' alpha_vs(vs) = min( alpha_vs(vs), alpha_tri); */
        for (nv = 0; nv < 3; nv++) {
          u0 = alpha_vs->data[tris->data[ii + tris->size[0] * nv] - 1];
          u0 = u0 <= alpha_tri ? u0 : alpha_tri;
          minval[nv] = u0;
        }

        for (i17 = 0; i17 < 3; i17++) {
          alpha_vs->data[tris->data[ii + tris->size[0] * i17] - 1] = minval[i17];
        }
      }
    }
  }
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
 * function [us_smooth,scaled] = smoothing_single_iteration(nv_clean, xs, tris,...
 *     nrms, opphes, isridge, ridgeedge, flabel, scaled, min_angle, angletol_min,...
 *     refareas2, mu, check_trank, hisurf_args, verbose)
 */
static void smoothing_single_iteration(int32_T nv_clean, const emxArray_real_T
  *xs, const emxArray_int32_T *tris, const emxArray_real_T *nrms, const
  emxArray_int32_T *opphes, const emxArray_boolean_T *isridge, const
  emxArray_boolean_T *ridgeedge, const emxArray_int32_T *flabel, boolean_T
  *scaled, real_T min_angle, boolean_T check_trank, const char_T
  hisurf_args_method[3], int32_T hisurf_args_degree, int32_T verbose,
  emxArray_real_T *us_smooth)
{
  int32_T count;
  boolean_T exitg1;
  boolean_T b_scaled;

  /* 'smoothing_single_iteration:4' coder.extrinsic('warning') */
  /* 'smoothing_single_iteration:5' coder.inline('never') */
  /* % Smoothing iteration */
  /* 'smoothing_single_iteration:7' tol = 0.1; */
  /* 'smoothing_single_iteration:8' if scaled || min_angle < angletol_min */
  if ((*scaled) || (min_angle < 5.0)) {
    /*  Step 1: Weighted Laplacian Smoothing */
    /* 'smoothing_single_iteration:10' us_smooth = weighted_Laplacian_tri_cleanmesh(nv_clean,xs, tris, isridge, ridgeedge, flabel,check_trank); */
    c_weighted_Laplacian_tri_cleanm(nv_clean, xs, tris, isridge, ridgeedge,
      flabel, check_trank, us_smooth);

    /*  Step 2: */
    /* 'smoothing_single_iteration:13' us_smooth = adjust_disps_onto_hisurf_cleanmesh(nv_clean, xs, us_smooth, nrms, ... */
    /* 'smoothing_single_iteration:14'         tris, opphes, isridge, ridgeedge, flabel, hisurf_args); */
    c_adjust_disps_onto_hisurf_clea(nv_clean, xs, us_smooth, nrms, tris, opphes,
      hisurf_args_method, hisurf_args_degree);

    /*  Step 3:(a) Update variable "us_smooth" of "nv_clean" pnts */
    /*  (b) Communicate variable "us_smooth" of ghost pnts (>nv_clean) */
  } else {
    /* 'smoothing_single_iteration:18' else */
    /*  Step 4: Isometric Smoothing */
    /* 'smoothing_single_iteration:20' us_smooth = ismooth_trimesh_cleanmesh(nv_clean,xs, tris, isridge, flabel, refareas2, mu, check_trank); */
    ismooth_trimesh_cleanmesh(nv_clean, xs, tris, isridge, flabel, check_trank,
      us_smooth);

    /* us_smooth = onering_scale_disps_tri_cleanmesh(nv_clean, xs, tris, nrms, us_smooth, opphes); */
    /*  Step 5:(a) Update variable "us_smooth" of "nv_clean" pnts */
    /*  (b) Communicate variable "us_smooth" of ghost pnts (>nv_clean) */
    /*  Step 6: Asynchronously rescale tangential displacements. */
    /* 'smoothing_single_iteration:27' [us_smooth,escaled] = async_scale_disps_tri_cleanmesh(nv_clean, xs, us_smooth, tris, tol); */
    async_scale_disps_tri_cleanmesh(nv_clean, xs, us_smooth, tris);

    /*  Adjust the displacements onto the high order surface */
    /* 'smoothing_single_iteration:30' scaled = true; */
    *scaled = TRUE;

    /* 'smoothing_single_iteration:30' count=int32(0); */
    count = 0;

    /* 'smoothing_single_iteration:31' while scaled */
    exitg1 = 0U;
    while ((exitg1 == 0U) && (*scaled)) {
      /*  Step 7: Adjusting displacements onto the high order surface */
      /* 'smoothing_single_iteration:33' us_smooth = adjust_disps_onto_hisurf_cleanmesh(nv_clean, xs, us_smooth, nrms, ... */
      /* 'smoothing_single_iteration:34'             tris, opphes, isridge, ridgeedge, flabel, hisurf_args); */
      c_adjust_disps_onto_hisurf_clea(nv_clean, xs, us_smooth, nrms, tris,
        opphes, hisurf_args_method, hisurf_args_degree);

      /*  Step 8:(a) Update variable "us_smooth" of "nv_clean" pnts */
      /*  (b) Communicate variable "us_smooth" of ghost pnts (>nv_clean) */
      /*  Step 9: Asynchronously rescale tangential displacements. */
      /* 'smoothing_single_iteration:40' [us_smooth,scaled] = async_scale_disps_tri_cleanmesh(nv_clean, xs, us_smooth, tris, tol); */
      b_scaled = async_scale_disps_tri_cleanmesh(nv_clean, xs, us_smooth, tris);
      *scaled = b_scaled;

      /* 'smoothing_single_iteration:41' if ~scaled */
      if (!b_scaled) {
        exitg1 = 1U;
      } else {
        /* 'smoothing_single_iteration:43' else */
        /* 'smoothing_single_iteration:44' count = count+1; */
        count++;

        /* 'smoothing_single_iteration:45' if count>4 */
        if (count > 4) {
          /* 'smoothing_single_iteration:46' if verbose>1 */
          exitg1 = 1U;
        }
      }
    }
  }
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

emxArray_boolean_T *emxCreateND_boolean_T(int32_T numDimensions, int32_T *size)
{
  emxArray_boolean_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  b_emxInit_boolean_T(&emx, numDimensions);
  numEl = 1;
  loop_ub = numDimensions - 1;
  for (i = 0; i <= loop_ub; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (boolean_T *)calloc((uint32_T)numEl, sizeof(boolean_T));
  emx->numDimensions = numDimensions;
  emx->allocatedSize = numEl;
  return emx;
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
  d_emxInit_real_T(&emx, numDimensions);
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

emxArray_boolean_T *emxCreateWrapperND_boolean_T(boolean_T *data, int32_T
  numDimensions, int32_T *size)
{
  emxArray_boolean_T *emx;
  int32_T numEl;
  int32_T loop_ub;
  int32_T i;
  b_emxInit_boolean_T(&emx, numDimensions);
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
  d_emxInit_real_T(&emx, numDimensions);
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

emxArray_boolean_T *emxCreateWrapper_boolean_T(boolean_T *data, int32_T rows,
  int32_T cols)
{
  emxArray_boolean_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  b_emxInit_boolean_T(&emx, 2);
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
  d_emxInit_real_T(&emx, 2);
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

emxArray_boolean_T *emxCreate_boolean_T(int32_T rows, int32_T cols)
{
  emxArray_boolean_T *emx;
  int32_T size[2];
  int32_T numEl;
  int32_T i;
  size[0] = rows;
  size[1] = cols;
  b_emxInit_boolean_T(&emx, 2);
  numEl = 1;
  for (i = 0; i < 2; i++) {
    numEl *= size[i];
    emx->size[i] = size[i];
  }

  emx->data = (boolean_T *)calloc((uint32_T)numEl, sizeof(boolean_T));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
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
  d_emxInit_real_T(&emx, 2);
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

void emxDestroyArray_boolean_T(emxArray_boolean_T *emxArray)
{
  emxFree_boolean_T(&emxArray);
}

void emxDestroyArray_int32_T(emxArray_int32_T *emxArray)
{
  emxFree_int32_T(&emxArray);
}

void emxDestroyArray_real_T(emxArray_real_T *emxArray)
{
  emxFree_real_T(&emxArray);
}

/*
 * function [xs, tris] = smooth_mesh_hisurf_cleanmesh( nv_clean, nt_clean, xs, ...
 * tris, degree, isridge, ridgeedge, flabel, niter, verbose, check_trank)
 */
void smooth_mesh_hisurf_cleanmesh(int32_T nv_clean, int32_T nt_clean,
  emxArray_real_T *xs, const emxArray_int32_T *tris, int32_T degree, const
  emxArray_boolean_T *isridge, const emxArray_boolean_T *ridgeedge, const
  emxArray_int32_T *flabel, int32_T niter, int32_T verbose, boolean_T
  check_trank)
{
  int32_T step;
  static const char_T cv8[3] = { 'C', 'M', 'F' };

  char_T hisurf_args_method[3];
  emxArray_real_T *nrms;
  emxArray_int32_T *opphes;
  real_T max_area;
  real_T min_area;
  real_T max_angle;
  real_T min_angle;
  boolean_T scaled;
  emxArray_real_T *us_smooth;
  boolean_T exitg1;
  boolean_T guard1 = FALSE;

  /* % */
  /* 'smooth_mesh_hisurf_cleanmesh:7' coder.extrinsic('fprintf') */
  /* 'smooth_mesh_hisurf_cleanmesh:8' if nargin<6 */
  /* 'smooth_mesh_hisurf_cleanmesh:9' if nargin<7 */
  /* 'smooth_mesh_hisurf_cleanmesh:10' if nargin<8 */
  /* 'smooth_mesh_hisurf_cleanmesh:11' if nargin<9 */
  /* 'smooth_mesh_hisurf_cleanmesh:12' if nargin<10 */
  /* 'smooth_mesh_hisurf_cleanmesh:13' if nargin<11 */
  /* 'smooth_mesh_hisurf_cleanmesh:15' hisurf_args.method = 'CMF'; */
  for (step = 0; step < 3; step++) {
    hisurf_args_method[step] = cv8[step];
  }

  emxInit_real_T(&nrms, 2);
  b_emxInit_int32_T(&opphes, 2);

  /* 'smooth_mesh_hisurf_cleanmesh:16' hisurf_args.degree = degree; */
  /* 'smooth_mesh_hisurf_cleanmesh:17' refareas2 = zeros(0,1); */
  /* 'smooth_mesh_hisurf_cleanmesh:18' mu = 0; */
  /* 'smooth_mesh_hisurf_cleanmesh:19' angletol_min = 5; */
  /* 'smooth_mesh_hisurf_cleanmesh:20' angletol_max = 27; */
  /* % Compute the following for the input clean mesh : 1. Normals, 2. Opposite half edges, 3. Quality */
  /* 'smooth_mesh_hisurf_cleanmesh:23' nrms = compute_hisurf_normals(nv_clean, xs, tris, degree); */
  compute_hisurf_normals(nv_clean, xs, tris, degree, nrms);

  /* 'smooth_mesh_hisurf_cleanmesh:24' opphes = determine_opposite_halfedge( int32(size(xs,1)), tris); */
  determine_opposite_halfedge(xs->size[0], tris, opphes);

  /* 'smooth_mesh_hisurf_cleanmesh:25' [min_angle, max_angle, min_area, max_area] = compute_statistics_tris_global(nt_clean, xs, tris); */
  compute_statistics_tris_global(nt_clean, xs, tris, &min_angle, &max_angle,
    &min_area, &max_area);

  /* 'smooth_mesh_hisurf_cleanmesh:26' if verbose */
  /* 'smooth_mesh_hisurf_cleanmesh:30' min_angle_pre = min_angle; */
  /* 'smooth_mesh_hisurf_cleanmesh:31' scaled = false; */
  scaled = FALSE;

  /* % Smooth mesh */
  /* 'smooth_mesh_hisurf_cleanmesh:34' for step=1:niter */
  step = 1;
  emxInit_real_T(&us_smooth, 2);
  exitg1 = 0U;
  while ((exitg1 == 0U) && (step <= niter)) {
    /* 'smooth_mesh_hisurf_cleanmesh:35' if verbose>1 */
    /* 'smooth_mesh_hisurf_cleanmesh:39' [us_smooth,scaled] = smoothing_single_iteration(nv_clean, xs, tris,... */
    /* 'smooth_mesh_hisurf_cleanmesh:40'         nrms, opphes, isridge, ridgeedge, flabel, scaled, min_angle, angletol_min,... */
    /* 'smooth_mesh_hisurf_cleanmesh:41'         refareas2, mu, check_trank, hisurf_args, verbose); */
    smoothing_single_iteration(nv_clean, xs, tris, nrms, opphes, isridge,
      ridgeedge, flabel, &scaled, min_angle, check_trank, hisurf_args_method,
      degree, verbose, us_smooth);

    /* 'smooth_mesh_hisurf_cleanmesh:44' if scaled */
    guard1 = FALSE;
    if (scaled) {
      guard1 = TRUE;
    } else {
      /* 'smooth_mesh_hisurf_cleanmesh:45' [pnt_added,xs] = add_disps_to_pntpositions(nv_clean, nt_clean, xs, tris, ... */
      /* 'smooth_mesh_hisurf_cleanmesh:46'     us_smooth, min_angle_pre, angletol_max, scaled); */
      scaled = add_disps_to_pntpositions(nv_clean, nt_clean, xs, tris, us_smooth,
        min_angle, FALSE);

      /* 'smooth_mesh_hisurf_cleanmesh:48' if pnt_added */
      if (scaled) {
        /*  Step 1: (a) Update position "xs" of "nv_clean" pnts */
        /*  (b) Communicate position "xs" of ghost pnts (>nv_clean) */
        /*  Step 2: Compute the normals for the new mesh */
        /* 'smooth_mesh_hisurf_cleanmesh:53' nrms = compute_hisurf_normals(nv_clean,xs,tris, hisurf_args.degree); */
        compute_hisurf_normals(nv_clean, xs, tris, degree, nrms);

        /* 'smooth_mesh_hisurf_cleanmesh:58' [min_angle, max_angle, min_area, max_area] = compute_statistics_tris_global(nt_clean, xs, tris); */
        compute_statistics_tris_global(nt_clean, xs, tris, &min_angle,
          &max_angle, &min_area, &max_area);

        /* 'smooth_mesh_hisurf_cleanmesh:60' if verbose>1 */
        /* 'smooth_mesh_hisurf_cleanmesh:64' min_angle_pre = min_angle; */
        /* 'smooth_mesh_hisurf_cleanmesh:64' scaled = false; */
        scaled = FALSE;
        guard1 = TRUE;
      } else {
        /* 'smooth_mesh_hisurf_cleanmesh:54' else */
        exitg1 = 1U;
      }
    }

    if (guard1 == TRUE) {
      step++;
    }
  }

  emxFree_real_T(&us_smooth);
  emxFree_int32_T(&opphes);
  emxFree_real_T(&nrms);
}
