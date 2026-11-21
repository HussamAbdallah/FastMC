/*
 * numpaif.h
 * external interface to spice frontend  subckt.c
 */

#ifndef ngspice_NUMPAIF_H
#define ngspice_NUMPAIF_H

#define  NUPADECKCOPY 0
#define  NUPASUBSTART 1
#define  NUPASUBDONE  2
#define  NUPAEVALDONE 3

struct card;

extern char  *nupa_copy(struct card *c);
extern int    nupa_eval(struct card *card);
extern void   nupa_signal(int sig);
extern void   nupa_scan(const struct card *card);
extern void   nupa_list_params(FILE *cp_out);
extern double nupa_get_param(const char *param_name, int *found);
extern const char *nupa_get_string_param(const char *param_name);
extern void   nupa_add_param(char *param_name, double value);
extern void   nupa_copy_inst_dico(void);
extern void   nupa_del_dicoS(void);
extern int    nupa_add_dicoslist(void);
extern void   nupa_rem_dicoslist(int);
extern void   nupa_set_dicoslist(int);

extern int dynMaxckt; /* number of lines in deck after expansion */

/* MC agauss sigma-capture API (used by mc_params_only mode in runcoms.c) */
extern void        mc_agauss_start_capture(void);  /* also frees any previous buffer */
extern void        mc_agauss_free_buf(void);        /* release buffer and all name strings */
extern int         mc_agauss_end_capture(void);
extern int         mc_agauss_get_count(void);
extern double      mc_agauss_get_draw(int i);
extern const char *mc_agauss_get_name(int i);
extern double      mc_agauss_get_nominal(int i);
extern double      mc_agauss_get_abs_var(int i);
extern double      mc_agauss_get_sigma(int i);
extern void        mc_agauss_redraw(void);
/* Deterministic seeded redraw: srand(seed_index) + TausSeed() + N x gauss1().
 * Every call with the same seed_index produces the same draws, making any row
 * in mc_params.mc independently reproducible for a full-simulation replay. */
extern void        mc_agauss_redraw_seeded(int seed_index);
/* Replay mode: agauss() returns pre-stored draws in buffer order instead of
 * calling gauss1().  Used with monte_seed=K to reproduce a specific seed's
 * parameter set for a full circuit simulation. */
extern void        mc_agauss_start_replay(void);
extern void        mc_agauss_end_replay(void);
/* Register PDK agauss funcname→(abs_var,sigma) so agauss() can append the
 * parameter name to the instance path (e.g. "x1.x0.x1" → "x1.x0.x1.mis_vth").
 * Called from inpcom.c inp_new_func() for every agauss .func entry. */
extern void        mc_agauss_register_func(const char *funcname, double abs_var, double sigma);
extern void        mc_agauss_clear_func_table(void);

#endif
