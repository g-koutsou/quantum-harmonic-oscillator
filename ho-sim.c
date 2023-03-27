#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <argp.h>

/* Auxiliary vectors */
#define MAX_AUX 3
double *aux[MAX_AUX];

/***
 * Return a random number in [0, 1)
 ***/
double
urand(void)
{
  double x = (double)rand()/(double)RAND_MAX;
  return x;
}

/***
 * Open file; print error if file pointer returned is NULL
 ***/
FILE *
uopen(const char *path, const char *mode)
{
  FILE *fp = fopen(path, mode);
  if(fp == NULL) {
    fprintf(stderr, "fopen(\"%s\", \'%s\') returned null; quitting...\n", path, mode);
    exit(-3);
  }
  return fp;
}

/***
 * Allocate memory with minimal error detection
 ***/
void *
ualloc(size_t size)
{
  void *ptr = malloc(size);
  if(ptr == NULL) {
    fprintf(stderr, "malloc() returned null; quitting...\n");
    exit(-2);
  }
  return ptr;
}

/***
 * Structure for storing action parameters
 ***/
typedef struct {
  double ma, om, alat, mass, omega, T;
  int N;
} action_params;

/***
 * Potential energy of path x[N], given number of points (N)  and rescaled mass
 * `ma' and frequency `om'
 ***/
double
E(action_params act, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;
  double e = 0;
  for(int i=0; i<N; i++) {
    e += x[i]*x[i];
  }
  return e*0.5*ma*om*om;
}

/***
 * Action of a path x[N], given number of points (N) and rescaled mass
 * `ma' and frequency `om'
 ***/
double
S(action_params act, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;
  double s = 0;
  for(int i=0; i<N; i++) {
    int ip = (i+1) % N;
    double dx = (x[ip] - x[i]);
    s += 0.5*ma*(dx*dx + om*om*x[i]*x[i]);
  }
  return s;
}

/***
 * Difference of action between paths x'[N] and x[N], assuming x'[k] =
 * x[k] for all k except k == i in which case x'[i] = xt
 ***/
double
dS(action_params act, int i, double xt, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;  
  int ip = (i+1)%N;
  int im = (N+i-1)%N;
  double dx2 = (xt*xt - x[i]*x[i]);
  double xp = x[ip];
  double xm = x[im];
  double ds = ma*dx2*(1 + om*om/2.0) - ma*(xp + xm)*(xt - x[i]);
  return ds;
}

/***
 * return N normally distributed random numbers in x[] with mean mu
 * and standard deviation si. x[] must already be allocated.
 ***/
void
normal(int N, double *x, double mu, double si)
{
  for(int i=0; i<N; i++) {
    double u0 = urand();
    double u1 = urand();
    double z = sqrt(-2.0*log(u0))*cos(2*M_PI*u1);
    x[i] = si*z + mu;
  }
  return;
}

/***
 * perform a metropolis sweep and return the acceptance rate
 ***/
double
metropolis_sweep(action_params act, double delta, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;
  double xt;
  double acc = 0;
  for(int j=0; j<N; j++) {
    normal(1, &xt, x[j], delta);
    double d = dS(act, j, xt, x);
    if(d < 0) {
      x[j] = xt;
      acc += 1;
    } else if (urand() < exp(-d)) {
      x[j] = xt;
      acc += 1;
    }
  }
  acc /= N;
  return acc;
}

/***
 * Hamiltonian given pseudo-momenta `pi' and trajectory `x'
 ***/
double
H(action_params act, double *pi, double *x)
{
  double h = S(act, x);
  for(int i=0; i<act.N; i++)
    h += 0.5*pi[i]*pi[i];
  return h;
}

/***
 * Force, defined as -\frac{\delta H}{\delta x_k}. `fo' should already
 * be allocated
 ***/
void
force(double *fo, action_params act, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;
  for(int i=0; i<N; i++) {
    int ip = (i+1)%N;
    int im = (N+i-1)%N;
    double xp = x[ip];
    double xm = x[im];
    fo[i] = (xp + xm - x[i]*(2+om*om))*ma;
  }
  return;
}

/***
 * Move field `out' by `tau' times `in'
 ***/
void
move_field(double *out, action_params act, double tau, double *in)
{
  int N = act.N;
  for(int i=0; i<N; i++) {
    out[i] += tau*in[i];
  }
  return;
}

/***
 * Initialize temporary vectors needed for HMC
 ***/
void
hmc_init(action_params act)
{
  int N = act.N;
  double *fo = ualloc(sizeof(double)*N);
  double *x0 = ualloc(sizeof(double)*N);
  double *pi = ualloc(sizeof(double)*N);

  aux[0] = x0;
  aux[1] = pi;
  aux[2] = fo;
  return;
}

/***
 * Free temporary vectors needed for HMC
 ***/
void
hmc_finalize()
{
  free(aux[0]);
  free(aux[1]);
  free(aux[2]);
  return;
}

/***
 * perform an HMC integration and return the whether accepted (1) or not (0)
 ***/
int
hmc(action_params act, int n_tau, double tau, double *x)
{
  double ma = act.ma;
  double om = act.om;
  int N = act.N;
  int acc;

  double *x0 = aux[0];
  double *pi = aux[1];
  double *fo = aux[2];
  
  /* Backup x */
  memcpy(x0, x, sizeof(double)*N);
  
  /* Pseudo-momenta */
  normal(N, pi, 0, 1);

  double h0 = H(act, pi, x);

  /* Force */  
  force(fo, act, x);

  /* Initial momenta half-step */
  move_field(pi, act, tau/2, fo);

  /* Do last iteration separately */
  for(int i=1; i<n_tau; i++) {
    move_field(x, act, tau, pi);
    force(fo, act, x);
    move_field(pi, act, tau, fo);
  }

  /* Last iteration with momenta half-step */
  move_field(x, act, tau, pi);
  force(fo, act, x);
  move_field(pi, act, tau/2, fo);
  
  /* Metropolis accept/reject */
  double h1 = H(act, pi, x);
  double dH = h1-h0;
  if(dH < 0) {
    acc = 1;
  } else if(urand() < exp(-dH)) {
    acc = 1;
  } else {
    memcpy(x, x0, sizeof(double)*N);
    acc = 0;
  }
  
  return acc;
}

/***
 * Append x[N] in a single line at the end of the text file pointed to by *fp 
 ***/
void
writex(FILE *fp, action_params act, double *x)
{
  int N = act.N;
  double alat = act.alat;
  for(int i=0; i<N; i++)
    fprintf(fp, " %+8.4e", alat*x[i]);
  fprintf(fp, "\n");
  return;
}

static char doc[] = "\n"
  "Quantum harmonic oscilator";

static char args_doc[] = "metr" "\n" "hmc";

static struct argp_option options[] =
  {
   {"output",       'o', "F",      0,  "Output to file F"},
   {"alat",         'a', "A",      0,  "Set lattice spacing to A"},
   {"time",         't', "T",      0,  "Set time-extent to T"},
   {"ntraj",        'n', "N",      0,  "Run for N trajectories in total"},
   {"mass",         'm', "M",      0,  "Mass parameter of action"},
   {"freq",         'w', "W",      0,  "Frequency (omega) parameter of action"},
   {"store-every",  's', "N",      0,  "Store every N trajectories"},
   {"print-every",  'p', "N",      0,  "Print info to screen every N trajectories"},
   {"ntherm",       'i', "N",      0,  "Use this many trajectories to calibrate delta (metr only)"},
   {"delta",        'd', "D",      0,  "Starting width of gaussian to draw trials (metr only)"},   
   {"target-accept",'r', "R",      0,  "Tune delta to achieve this target accpetance rate (metr only)"},
   {"tau",          'u', "D",      0,  "HMC integration step (hmc only)"},
   {"n-tau",        'l', "D",      0,  "Number of HMC integration steps (hmc only)"},   
   { 0 }
  };

struct arguments
{
  enum {HMC, METR, NONE} mode;
  char *output;
  double alat, T, mass, omega;
  int ntraj, store_every, print_every;
  double delta, target_acc, tau;
  int ntherm, n_tau;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state);
static struct argp argp = {options, parse_opt, args_doc, doc};

static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
  struct arguments *arguments = state->input;
  switch (key) {
  case 'o':
    arguments->output = arg;
    break;
  case 'a':
    arguments->alat = strtod(arg, NULL);
    break;
  case 'm':
    arguments->mass = strtod(arg, NULL);
    break;
  case 'w':
    arguments->omega = strtod(arg, NULL);
    break;
  case 't':
    arguments->T = strtod(arg, NULL);
    break;
  case 'n':
    arguments->ntraj = strtoul(arg, NULL, 10);
    break;
  case 's':
    arguments->store_every = strtoul(arg, NULL, 10);
    break;
  case 'p':
    arguments->print_every = strtoul(arg, NULL, 10);
    break;
    /* Complain if the following are specified within hmc
       subcommand */
  case 'd':
    if(arguments->mode != METR) {
      argp_error(state, "%s: option should only be specified with \"metr\"", state->argv[state->next-2]);
    }
    arguments->delta = strtod(arg, NULL);
    break;
  case 'r':
    if(arguments->mode != METR) {
      argp_error(state, "%s: option should only be specified with \"metr\"", state->argv[state->next-2]);
    }
    arguments->target_acc = strtod(arg, NULL);
    break;
  case 'i':
    if(arguments->mode != METR) {
      argp_error(state, "%s: option should only be specified with \"metr\"", state->argv[state->next-2]);
    }
    arguments->ntherm = strtoul(arg, NULL, 10);
    break;    
    /* Complain if the following are specified within metr
       subcommand */
  case 'u':
    if(arguments->mode != HMC) {
      argp_error(state, "%s: option should only be specified with \"hmc\"", state->argv[state->next-2]);
    }
    arguments->tau = strtod(arg, NULL);
    break;
  case 'l':
    if(arguments->mode != HMC) {
      argp_error(state, "%s: option should only be specified with \"hmc\"", state->argv[state->next-2]);
    }
    arguments->n_tau = strtoul(arg, NULL, 10);
    break;
  case ARGP_KEY_ARG:
    if (state->arg_num != 0)
      /* Too many arguments */
      argp_usage (state);
    if(strcmp(arg, "metr") == 0) {
      arguments->mode = METR;
    } else if(strcmp(arg, "hmc") == 0) {
      arguments->mode = HMC;
    } else {
      argp_error(state, "%s: wrong subcommand; should be either \"metr\" or \"hmc\"", arg);
    }
    break;
  case ARGP_KEY_END:
    if (state->arg_num != 1)
      /* Not enough arguments. */
      argp_usage(state);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static error_t
parse_opt_metr(int key, char *arg, struct argp_state *state)
{
  struct arguments_metr *arguments = state->input;  
  switch (key) {
  case ARGP_KEY_END:
    if (state->arg_num != 1)
      /* Not enough arguments. */
      argp_usage(state);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

int
main(int argc, char *argv[])
{
  struct arguments arguments;

  arguments.mode = NONE;
  arguments.ntraj = 1000;
  arguments.store_every = 1;
  arguments.print_every = 1;
  arguments.T = 1;
  arguments.alat = 0.01;
  arguments.mass = 1;
  arguments.omega = 1;
  arguments.output = "x.txt";
  
  arguments.ntherm = 0;
  arguments.delta = 0.1;
  arguments.target_acc = 0.5;

  arguments.tau = 0.5;
  arguments.n_tau = 2;

  /*
    This first call with argc set to 2 forces the mode to be parsed
    and set first, so that the mode is known when parsing options
   */
  argp_parse(&argp, 2, argv, 0, 0, &arguments);
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  int mode = arguments.mode;

  action_params act;
  act.T = arguments.T;
  act.alat = arguments.alat;
  act.mass = arguments.mass;
  act.omega = arguments.omega;
  int ntraj = arguments.ntraj;
  int store_every = arguments.store_every;
  int print_every = arguments.print_every;
  act.N = act.T/act.alat;
  int N = act.N;
  double *x = ualloc(sizeof(double)*N);

  /* metropolis parameters */
  double delta = arguments.delta;
  int ntherm = arguments.ntherm;
  double target_acc = arguments.target_acc;

  /* HMC parameters */
  double tau = arguments.tau;
  int n_tau = arguments.n_tau;

  /* Rescaled parameters */
  act.ma = act.mass*act.alat;
  act.om = act.omega*act.alat;
  
  FILE *fp = uopen(arguments.output, "w");
  normal(N, x, 0, delta);

  if(mode == HMC)
    hmc_init(act);
  
  int acc = 0;
  for(int i=0; i<ntraj+ntherm; i++) {
    double r_acc;
    if(mode == METR) {
      r_acc = metropolis_sweep(act, delta, x);
      if(i < ntherm) {
	if(acc > target_acc)
	  delta *= 1.025;
	if(acc < target_acc)
	  delta *= 0.975;
      }
    } else if(mode == HMC) {
      acc += hmc(act, n_tau, tau, x);
      r_acc = acc / (double)(i+1);
    }

    if(i % print_every == 0) {
      printf(" i = %8d | a = %6.4f | S(x) = %12.4f | E(x) = %4.2e | racc = %4.2f",
	     i, act.alat, S(act, x), E(act, x), r_acc);
      if(mode == METR) {
	printf("| delta = %5.3e\n", delta);
      } else {
	printf("\n");
      }
    }

    
    if(i % store_every == 0)
      writex(fp, act, x);
  }
  if(mode == HMC)
    hmc_finalize();

  
  free(x);
  fclose(fp);
  return 0;
}
  
