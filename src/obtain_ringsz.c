#include "util.h"

static void b_emxInit_int32_T(emxArray_int32_T **pEmxArray, int32_T
  numDimensions);
static void b_fix(real_T *x);

static int32_T b_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], const emxArray_boolean_T *vtags,
  const emxArray_boolean_T *ftags);

/* Function Definitions */
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
 * function [ngbvs, nverts, vtags, ftags, ngbfs, nfaces] = obtain_nring_surf...
 *     ( vid, ring, minpnts, tris, opphes, v2he, ngbvs, vtags, ftags, ngbfs)
 */
static int32_T b_obtain_nring_surf(int32_T vid, real_T ring, int32_T minpnts,
  const emxArray_int32_T *tris, const emxArray_int32_T *opphes, const
  emxArray_int32_T *v2he, int32_T ngbvs[128], const emxArray_boolean_T *vtags,
  const emxArray_boolean_T *ftags)
{
  int32_T nverts;
  emxArray_boolean_T *b_ftags;
  int32_T lid_prv;
  int32_T lid;
  emxArray_boolean_T *b_vtags;
  int32_T fid;
  int32_T nfaces;
  boolean_T overflow;
  boolean_T b0;
  int32_T fid_in;
  static const int8_T iv2[3] = { 2, 3, 1 };

  int32_T hebuf[128];
  int32_T exitg4;
  static const int8_T iv3[3] = { 3, 1, 2 };

  int32_T ngbfs[256];
  int32_T nverts_pre;
  int32_T nfaces_pre;
  real_T ring_full;
  real_T cur_ring;
  int32_T exitg1;
  boolean_T guard1 = FALSE;
  int32_T nverts_last;
  boolean_T exitg2;
  boolean_T isfirst;
  int32_T exitg3;
  boolean_T guard2 = FALSE;
  emxInit_boolean_T(&b_ftags, 1);
  lid_prv = b_ftags->size[0];
  b_ftags->size[0] = ftags->size[0];
  emxEnsureCapacity((emxArray__common *)b_ftags, lid_prv, (int32_T)sizeof
                    (boolean_T));
  lid = ftags->size[0] - 1;
  for (lid_prv = 0; lid_prv <= lid; lid_prv++) {
    b_ftags->data[lid_prv] = ftags->data[lid_prv];
  }

  emxInit_boolean_T(&b_vtags, 1);
  lid_prv = b_vtags->size[0];
  b_vtags->size[0] = vtags->size[0];
  emxEnsureCapacity((emxArray__common *)b_vtags, lid_prv, (int32_T)sizeof
                    (boolean_T));
  lid = vtags->size[0] - 1;
  for (lid_prv = 0; lid_prv <= lid; lid_prv++) {
    b_vtags->data[lid_prv] = vtags->data[lid_prv];
  }

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
      b0 = TRUE;
    } else {
      b0 = FALSE;
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
      ngbvs[0] = tris->data[fid + tris->size[0] * (iv2[lid] - 1)];

      /* 'obtain_nring_surf:99' if ~oneringonly */
      if (!b0) {
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
      lid_prv = iv3[lid] - 1;

      /* 'obtain_nring_surf:106' v = tris(fid, lid_prv); */
      /* 'obtain_nring_surf:108' if nverts<maxnv && nfaces<maxnf */
      if ((nverts < 128) && (nfaces < 256)) {
        /* 'obtain_nring_surf:109' nverts = nverts + 1; */
        nverts++;

        /* 'obtain_nring_surf:109' ngbvs( nverts) = v; */
        ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid_prv];

        /* 'obtain_nring_surf:111' if ~oneringonly */
        if (!b0) {
          /*  Save starting position for next vertex */
          /* 'obtain_nring_surf:113' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
          hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] * (iv3[lid_prv]
            - 1)];

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
      lid = opphes->data[fid + opphes->size[0] * lid_prv];

      /* 'obtain_nring_surf:121' fid = heid2fid(opp); */
      /*  HEID2FID   Obtains face ID from half-edge ID. */
      /* 'heid2fid:3' coder.inline('always'); */
      /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
      fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] * lid_prv] >>
                      2U) - 1;

      /* 'obtain_nring_surf:123' if fid == fid_in */
      if (fid + 1 == fid_in) {
        exitg4 = 1U;
      } else {
        /* 'obtain_nring_surf:125' else */
        /* 'obtain_nring_surf:126' lid = heid2leid(opp); */
        /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
        /* 'heid2leid:3' coder.inline('always'); */
        /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
        lid = (int32_T)((uint32_T)lid & 3U);
      }
    } while (exitg4 == 0U);

    /*  Finished cycle */
    /* 'obtain_nring_surf:130' if ring==1 && (nverts>=minpnts || nverts>=maxnv || nfaces>=maxnf || nargout<=2) */
    if (ring == 1.0) {
      /* 'obtain_nring_surf:131' if overflow */
    } else {
      /* 'obtain_nring_surf:137' vtags(vid) = true; */
      b_vtags->data[vid - 1] = TRUE;

      /* 'obtain_nring_surf:138' for i=1:nverts */
      for (lid_prv = 1; lid_prv <= nverts; lid_prv++) {
        /* 'obtain_nring_surf:138' vtags(ngbvs(i))=true; */
        b_vtags->data[ngbvs[lid_prv - 1] - 1] = TRUE;
      }

      /* 'obtain_nring_surf:139' for i=1:nfaces */
      for (lid_prv = 1; lid_prv <= nfaces; lid_prv++) {
        /* 'obtain_nring_surf:139' ftags(ngbfs(i))=true; */
        b_ftags->data[ngbfs[lid_prv - 1] - 1] = TRUE;
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
          fid_in = nfaces;

          /* 'obtain_nring_surf:153' nverts_last = nverts; */
          nverts_last = nverts;

          /* 'obtain_nring_surf:154' for ii = nfaces_pre+1 : nfaces_last */
          while (nfaces_pre + 1 <= fid_in) {
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
                   != 0) && (!b_ftags->data[fid])) {
                /* 'obtain_nring_surf:161' lid = heid2leid(oppe); */
                /*  HEID2LEID   Obtains local edge ID within a face from half-edge ID. */
                /* 'heid2leid:3' coder.inline('always'); */
                /* 'heid2leid:4' leid = int32(bitand(uint32(heid),3))+1; */
                lid = (int32_T)((uint32_T)opphes->data[(ngbfs[nfaces_pre] +
                  opphes->size[0] * lid) - 1] & 3U);

                /* 'obtain_nring_surf:162' v = tris( fid, prv(lid)); */
                /* 'obtain_nring_surf:164' overflow = overflow || ~vtags(v) && nverts>=length(ngbvs) || ... */
                /* 'obtain_nring_surf:165'                         ~ftags(fid) && nfaces>=length(ngbfs); */
                if (overflow || ((!b_vtags->data[tris->data[fid + tris->size[0] *
                                  (iv3[lid] - 1)] - 1]) && (nverts >= 128)) ||
                    ((!b_ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:166' if ~ftags(fid) && ~overflow */
                if ((!b_ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:167' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:167' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:168' ftags(fid) = true; */
                  b_ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:171' if ~vtags(v) && ~overflow */
                if ((!b_vtags->data[tris->data[fid + tris->size[0] * (iv3[lid] -
                      1)] - 1]) && (!overflow)) {
                  /* 'obtain_nring_surf:172' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:172' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv3[lid]
                    - 1)];

                  /* 'obtain_nring_surf:173' vtags(v) = true; */
                  b_vtags->data[tris->data[fid + tris->size[0] * (iv3[lid] - 1)]
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
          if ((nverts >= minpnts) || (nfaces >= 256) || (nfaces == fid_in)) {
            exitg1 = 1U;
          } else {
            /* 'obtain_nring_surf:182' else */
            /*  If needs to expand, then undo the last half ring */
            /* 'obtain_nring_surf:184' for i=nverts_last+1:nverts */
            for (lid_prv = nverts_last; lid_prv + 1 <= nverts; lid_prv++) {
              /* 'obtain_nring_surf:184' vtags(ngbvs(i)) = false; */
              b_vtags->data[ngbvs[lid_prv] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:185' nverts = nverts_last; */
            nverts = nverts_last;

            /* 'obtain_nring_surf:187' for i=nfaces_last+1:nfaces */
            for (lid_prv = fid_in; lid_prv + 1 <= nfaces; lid_prv++) {
              /* 'obtain_nring_surf:187' ftags(ngbfs(i)) = false; */
              b_ftags->data[ngbfs[lid_prv] - 1] = FALSE;
            }

            /* 'obtain_nring_surf:188' nfaces = nfaces_last; */
            nfaces = fid_in;
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
              b0 = TRUE;
            } else {
              b0 = FALSE;
            }

            /* 'obtain_nring_surf:200' if allow_early_term */
            if (b0) {
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
              if (overflow || ((!b_vtags->data[tris->data[fid + tris->size[0] *
                                (iv2[lid] - 1)] - 1]) && (nverts >= 128))) {
                overflow = TRUE;
              } else {
                overflow = FALSE;
              }

              /* 'obtain_nring_surf:212' if ~overflow */
              if (!overflow) {
                /* 'obtain_nring_surf:213' nverts = nverts + 1; */
                nverts++;

                /* 'obtain_nring_surf:213' ngbvs( nverts) = v; */
                ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * (iv2[lid] -
                  1)];

                /* 'obtain_nring_surf:213' vtags(v)=true; */
                b_vtags->data[tris->data[fid + tris->size[0] * (iv2[lid] - 1)] -
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
              lid_prv = iv3[lid] - 1;

              /*  Insert face into list */
              /* 'obtain_nring_surf:226' if ftags(fid) */
              guard2 = FALSE;
              if (b_ftags->data[fid]) {
                /* 'obtain_nring_surf:227' if allow_early_term && ~isfirst */
                if (b0 && (!isfirst)) {
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
                if (overflow || ((!b_vtags->data[tris->data[fid + tris->size[0] *
                                  lid_prv] - 1]) && (nverts >= 128)) ||
                    ((!b_ftags->data[fid]) && (nfaces >= 256))) {
                  overflow = TRUE;
                } else {
                  overflow = FALSE;
                }

                /* 'obtain_nring_surf:235' if ~vtags(v) && ~overflow */
                if ((!b_vtags->data[tris->data[fid + tris->size[0] * lid_prv] -
                     1]) && (!overflow)) {
                  /* 'obtain_nring_surf:236' nverts = nverts + 1; */
                  nverts++;

                  /* 'obtain_nring_surf:236' ngbvs( nverts) = v; */
                  ngbvs[nverts - 1] = tris->data[fid + tris->size[0] * lid_prv];

                  /* 'obtain_nring_surf:236' vtags(v)=true; */
                  b_vtags->data[tris->data[fid + tris->size[0] * lid_prv] - 1] =
                    TRUE;

                  /*  Save starting position for next ring */
                  /* 'obtain_nring_surf:239' hebuf(nverts) = opphes( fid, prv(lid_prv)); */
                  hebuf[nverts - 1] = opphes->data[fid + opphes->size[0] *
                    (iv3[lid_prv] - 1)];
                }

                /* 'obtain_nring_surf:242' if ~ftags(fid) && ~overflow */
                if ((!b_ftags->data[fid]) && (!overflow)) {
                  /* 'obtain_nring_surf:243' nfaces = nfaces + 1; */
                  nfaces++;

                  /* 'obtain_nring_surf:243' ngbfs( nfaces) = fid; */
                  ngbfs[nfaces - 1] = fid + 1;

                  /* 'obtain_nring_surf:243' ftags(fid)=true; */
                  b_ftags->data[fid] = TRUE;
                }

                /* 'obtain_nring_surf:245' isfirst = false; */
                isfirst = FALSE;
                guard2 = TRUE;
              }

              if (guard2 == TRUE) {
                /* 'obtain_nring_surf:248' opp = opphes(fid, lid_prv); */
                lid = opphes->data[fid + opphes->size[0] * lid_prv];

                /* 'obtain_nring_surf:249' fid = heid2fid(opp); */
                /*  HEID2FID   Obtains face ID from half-edge ID. */
                /* 'heid2fid:3' coder.inline('always'); */
                /* 'heid2fid:4' fid = int32(bitshift(uint32(heid), -2)); */
                fid = (int32_T)((uint32_T)opphes->data[fid + opphes->size[0] *
                                lid_prv] >> 2U) - 1;

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
                  lid = (int32_T)((uint32_T)lid & 3U);
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
      /* 'obtain_nring_surf:269' for i=1:nverts */
      /* 'obtain_nring_surf:270' if ~oneringonly */
      /* 'obtain_nring_surf:271' if overflow */
    }
  }

  emxFree_boolean_T(&b_ftags);
  emxFree_boolean_T(&b_vtags);
  return nverts;
}

/*
 * function [ring_sz] = obtain_ringsz_cleanmesh(nv_clean, part_bdry, xs, elems, degree)
 */
void obtain_ringsz_cleanmesh(int32_T nv_clean, const emxArray_int32_T *part_bdry,
  const emxArray_real_T *xs, const emxArray_int32_T *elems, int32_T degree,
  emxArray_real_T *ring_sz)
{
  emxArray_int32_T *opphes;
  emxArray_int32_T *v2he;
  int32_T nv;
  int32_T i0;
  int32_T nx;
  int32_T numpnts;
  emxArray_boolean_T *r0;
  emxArray_boolean_T *r1;
  int32_T ngbvs[128];
  int32_T unusedU0[128];
  int32_T loop_ub;
  int32_T nverts;
  uint32_T ii;
  emxArray_real_T *int_pnts;
  emxArray_real_T *idx;
  emxArray_boolean_T *x;
  emxArray_boolean_T *r2;
  emxArray_boolean_T *r3;
  int32_T num_ints;
  int32_T b_ii;
  uint32_T c_ii;
  boolean_T exitg2;
  boolean_T exitg1;
  emxInit_int32_T(&opphes, 2);
  b_emxInit_int32_T(&v2he, 1);

  /*  This function computes the required ring size for each point in the given */
  /*  surface mesh according to the degree and with sufficient number of points */
  /*  in the rings. The inputs */
  /*  nv: The number of points for which the ring size has to be found */
  /*  elems: The element list */
  /*  degree: The degree of fitting */
  /*  Output: "ring_sz" is the optimum ring sizes for "nv" points */
  /* #coder.typeof(0,[inf,3],[1,0]),coder.typeof(int32(0),[inf,3],[1,0]),int32(0)} */
  /* 'obtain_ringsz_cleanmesh:12' nv= size(xs,1); */
  nv = xs->size[0];

  /* 'obtain_ringsz_cleanmesh:13' opphes = determine_opposite_halfedge( nv, elems); */
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
  /* 'determine_opposite_halfedge:18' if nargin<3 */
  /* 'determine_opposite_halfedge:19' switch size(elems,2) */
  /* 'determine_opposite_halfedge:20' case {3,6} % tri */
  /*  tri */
  /* 'determine_opposite_halfedge:21' opphes = determine_opposite_halfedge_tri(nv, elems); */
  determine_opposite_halfedge_tri((real_T)nv, elems, opphes);

  /* 'obtain_ringsz_cleanmesh:14' v2he = determine_incident_halfedges( elems, opphes); */
  determine_incident_halfedges(elems, opphes, v2he);

  /* 'obtain_ringsz_cleanmesh:15' ngbvs = coder.nullcopy(zeros(128,1,'int32')); */
  /* 'obtain_ringsz_cleanmesh:16' vtags = false(nv, 1); */
  /* 'obtain_ringsz_cleanmesh:17' ftags = false(size(elems,1), 1); */
  /* 'obtain_ringsz_cleanmesh:19' min_ringsz = 2; */
  /* 'obtain_ringsz_cleanmesh:20' max_ringsz = 4; */
  /*  Initialize the ring sizes for each interior/overlay point in the submesh */
  /*  to the above computed standard ring size */
  /* 'obtain_ringsz_cleanmesh:24' ring_sz = coder.nullcopy(zeros(nv_clean,1)); */
  i0 = ring_sz->size[0];
  ring_sz->size[0] = nv_clean;
  emxEnsureCapacity((emxArray__common *)ring_sz, i0, (int32_T)sizeof(real_T));

  /* 'obtain_ringsz_cleanmesh:25' for i=1:nv_clean */
  for (nx = 1; nx <= nv_clean; nx++) {
    /* 'obtain_ringsz_cleanmesh:26' ring_sz(i) = min_ringsz; */
    ring_sz->data[nx - 1] = 2.0;
  }

  /*  Assign some minimum count of no. of points that should be included in the ring */
  /* 'obtain_ringsz_cleanmesh:30' numpnts = int32(0); */
  numpnts = 0;

  /* 'obtain_ringsz_cleanmesh:30' degree = max(1,min(6,degree)); */
  if (6 > degree) {
  } else {
    degree = 6;
  }

  if (1 < degree) {
  } else {
    degree = 1;
  }

  /* 'obtain_ringsz_cleanmesh:31' switch degree */
  switch (degree) {
   case 1:
    /* 'obtain_ringsz_cleanmesh:32' case 1 */
    /* 'obtain_ringsz_cleanmesh:33' numpnts = int32(7); */
    numpnts = 7;
    break;

   case 2:
    /* 'obtain_ringsz_cleanmesh:34' case 2 */
    /* 'obtain_ringsz_cleanmesh:35' numpnts = int32(12); */
    numpnts = 12;
    break;

   case 3:
    /* 'obtain_ringsz_cleanmesh:36' case 3 */
    /* 'obtain_ringsz_cleanmesh:37' numpnts = int32(19); */
    numpnts = 19;
    break;

   case 4:
    /* 'obtain_ringsz_cleanmesh:38' case 4 */
    /* 'obtain_ringsz_cleanmesh:39' numpnts = int32(28); */
    numpnts = 28;
    break;

   case 5:
    /* 'obtain_ringsz_cleanmesh:40' case 5 */
    /* 'obtain_ringsz_cleanmesh:41' numpnts = int32(38); */
    numpnts = 38;
    break;

   case 6:
    /* 'obtain_ringsz_cleanmesh:42' case 6 */
    /* 'obtain_ringsz_cleanmesh:43' numpnts = int32(49); */
    numpnts = 49;
    break;
  }

  /* 'obtain_ringsz_cleanmesh:45' minpnts = int32(0); */
  /* 'obtain_ringsz_cleanmesh:46' for i=1:nv_clean */
  nx = 1;
  emxInit_boolean_T(&r0, 1);
  emxInit_boolean_T(&r1, 1);
  while (nx <= nv_clean) {
    /* 'obtain_ringsz_cleanmesh:47' [~, nverts] = obtain_nring_surf(i, ring_sz(i), minpnts, ... */
    /* 'obtain_ringsz_cleanmesh:48'         elems, opphes, v2he, ngbvs, vtags, ftags); */
    memcpy((void *)&unusedU0[0], (void *)&ngbvs[0], sizeof(int32_T) << 7);
    i0 = r0->size[0];
    r0->size[0] = nv;
    emxEnsureCapacity((emxArray__common *)r0, i0, (int32_T)sizeof(boolean_T));
    loop_ub = nv - 1;
    for (i0 = 0; i0 <= loop_ub; i0++) {
      r0->data[i0] = FALSE;
    }

    i0 = r1->size[0];
    r1->size[0] = elems->size[0];
    emxEnsureCapacity((emxArray__common *)r1, i0, (int32_T)sizeof(boolean_T));
    loop_ub = elems->size[0] - 1;
    for (i0 = 0; i0 <= loop_ub; i0++) {
      r1->data[i0] = FALSE;
    }

    nverts = b_obtain_nring_surf(nx, ring_sz->data[nx - 1], 0, elems, opphes, v2he,
      unusedU0, r0, r1);

    /* 'obtain_ringsz_cleanmesh:49' if (nverts < numpnts) */
    if (nverts < numpnts) {
      /* 'obtain_ringsz_cleanmesh:50' ring_sz(i) = max_ringsz; */
      ring_sz->data[nx - 1] = 4.0;
    }

    nx++;
  }

  emxFree_boolean_T(&r1);
  emxFree_boolean_T(&r0);

  /*  Performs a 1-ring search for all points on the partition boundary and */
  /*  assigns consistent ring sizes. */
  /* 'obtain_ringsz_cleanmesh:56' for ii=1:size(part_bdry,1) */
  ii = 1U;
  emxInit_real_T(&int_pnts, 1);
  emxInit_real_T(&idx, 1);
  emxInit_boolean_T(&x, 1);
  emxInit_boolean_T(&r2, 1);
  emxInit_boolean_T(&r3, 1);
  while (ii <= (uint32_T)part_bdry->size[0]) {
    /* 'obtain_ringsz_cleanmesh:57' if (part_bdry(ii)~=0)&&(ring_sz(ii)==2) */
    if ((part_bdry->data[(int32_T)ii - 1] != 0) && (ring_sz->data[(int32_T)ii -
         1] == 2.0)) {
      /* 'obtain_ringsz_cleanmesh:58' [ngbvs, nverts] = obtain_nring_surf(int32(ii), 1, minpnts, ... */
      /* 'obtain_ringsz_cleanmesh:59'             elems, opphes, v2he, ngbvs, vtags, ftags); */
      i0 = r2->size[0];
      r2->size[0] = nv;
      emxEnsureCapacity((emxArray__common *)r2, i0, (int32_T)sizeof(boolean_T));
      loop_ub = nv - 1;
      for (i0 = 0; i0 <= loop_ub; i0++) {
        r2->data[i0] = FALSE;
      }

      i0 = r3->size[0];
      r3->size[0] = elems->size[0];
      emxEnsureCapacity((emxArray__common *)r3, i0, (int32_T)sizeof(boolean_T));
      loop_ub = elems->size[0] - 1;
      for (i0 = 0; i0 <= loop_ub; i0++) {
        r3->data[i0] = FALSE;
      }

      nverts = b_obtain_nring_surf((int32_T)ii, 1.0, 0, elems, opphes, v2he, ngbvs,
        r2, r3);

      /* 'obtain_ringsz_cleanmesh:60' [int_pnts,num_ints]=point_classification(ngbvs,nverts,part_bdry,nv_clean); */
      /* 'obtain_ringsz_cleanmesh:72' int_pnts = coder.nullcopy(zeros(nverts,1)); */
      i0 = int_pnts->size[0];
      int_pnts->size[0] = nverts;
      emxEnsureCapacity((emxArray__common *)int_pnts, i0, (int32_T)sizeof(real_T));

      /* 'obtain_ringsz_cleanmesh:73' numints = int32(1); */
      num_ints = 1;

      /* 'obtain_ringsz_cleanmesh:74' for ii=1:nverts */
      for (b_ii = 0; b_ii + 1 <= nverts; b_ii++) {
        /* 'obtain_ringsz_cleanmesh:75' cur_pnt = ngbvs(ii); */
        /* 'obtain_ringsz_cleanmesh:76' if (cur_pnt <= nv_clean) */
        if (ngbvs[b_ii] <= nv_clean) {
          /* 'obtain_ringsz_cleanmesh:77' idx = find(part_bdry == cur_pnt, 1); */
          i0 = x->size[0];
          x->size[0] = part_bdry->size[0];
          emxEnsureCapacity((emxArray__common *)x, i0, (int32_T)sizeof(boolean_T));
          loop_ub = part_bdry->size[0] - 1;
          for (i0 = 0; i0 <= loop_ub; i0++) {
            x->data[i0] = (part_bdry->data[i0] == ngbvs[b_ii]);
          }

          nx = x->size[0];
          numpnts = 0;
          i0 = idx->size[0];
          idx->size[0] = 1;
          emxEnsureCapacity((emxArray__common *)idx, i0, (int32_T)sizeof(real_T));
          c_ii = 1U;
          exitg2 = 0U;
          while ((exitg2 == 0U) && (c_ii <= (uint32_T)nx)) {
            if (x->data[(int32_T)c_ii - 1]) {
              numpnts = 1;
              idx->data[0] = (real_T)c_ii;
              exitg2 = 1U;
            } else {
              c_ii++;
            }
          }

          if (numpnts == 0) {
            i0 = idx->size[0];
            idx->size[0] = 0;
            emxEnsureCapacity((emxArray__common *)idx, i0, (int32_T)sizeof
                              (real_T));
          }

          /* 'obtain_ringsz_cleanmesh:78' if isempty(idx) */
          if (idx->size[0] == 0) {
            /* 'obtain_ringsz_cleanmesh:79' int_pnts(numints) = cur_pnt; */
            int_pnts->data[num_ints - 1] = (real_T)ngbvs[b_ii];

            /* 'obtain_ringsz_cleanmesh:80' numints = numints + 1; */
            num_ints++;
          }
        }
      }

      /* 'obtain_ringsz_cleanmesh:61' for jj=1:num_ints */
      nx = 1;
      exitg1 = 0U;
      while ((exitg1 == 0U) && (nx <= num_ints)) {
        /* 'obtain_ringsz_cleanmesh:62' if ring_sz(int_pnts(jj))==4 */
        if (ring_sz->data[(int32_T)int_pnts->data[nx - 1] - 1] == 4.0) {
          /* 'obtain_ringsz_cleanmesh:63' ring_sz(ii) = 3; */
          ring_sz->data[(int32_T)ii - 1] = 3.0;
          exitg1 = 1U;
        } else {
          nx++;
        }
      }
    }

    ii++;
  }

  emxFree_boolean_T(&r3);
  emxFree_boolean_T(&r2);
  emxFree_boolean_T(&x);
  emxFree_real_T(&idx);
  emxFree_real_T(&int_pnts);
  emxFree_int32_T(&v2he);
  emxFree_int32_T(&opphes);
}
