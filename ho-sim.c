#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <argp.h>

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
 * Potential energy of path x[N], given number of points (N)  and rescaled mass
 * `ma' and frequency `om'
 ***/
double
E(int N, double ma, double om, double *x)
{
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
S(int N, double ma, double om, double *x)
{
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
dS(int N, double ma, double om, int i, double xt, double *x)
{
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
 * Append x[N] in a single line at the end of the text file pointed to by *fp 
 ***/
void
writex(FILE *fp, int N, double alat, double *x)
{
  for(int i=0; i<N; i++)
    fprintf(fp, " %+8.4e", alat*x[i]);
  fprintf(fp, "\n");
  return;
}

static char doc[] =
  "Quantum harmonic oscilator";

static char args_doc[] = "";

static struct argp_option options[] =
  {
   {"output",       'o', "F",      0,  "Output to file F"},
   {"alat",         'a', "A",      0,  "Set lattice spacing to A"},
   {"time",         't', "T",      0,  "Set time-extent to T"},
   {"ntraj",        'n', "N",      0,  "Run for N trajectories in total"},
   {"ntherm",       'i', "N",      0,  "Use this many trajectories to calibrate delta"},
   {"mass",         'm', "M",      0,  "Mass parameter of action"},
   {"freq",         'w', "W",      0,  "Frequency (omega) parameter of action"},
   {"store-every",  's', "N",      0,  "Store every N trajectories"},
   {"print-every",  'p', "N",      0,  "Print info to screen every N trajectories"},
   {"delta",        'd', "D",      0,  "Starting width of gaussian to draw trials"},   
   {"target-accept",'r', "R",      0,  "Tune delta to achieve this target accpetance rate"},
   { 0 }
  };

struct arguments
{
  char *output;
  double alat, T, delta, target_acc, mass, omega;
  int ntraj, store_every, print_every, ntherm;
};

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
  case 'd':
    arguments->delta = strtod(arg, NULL);
    break;
  case 'r':
    arguments->target_acc = strtod(arg, NULL);
    break;
  case 'i':
    arguments->ntherm = strtoul(arg, NULL, 10);
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
  case ARGP_KEY_END:
    if (state->arg_num != 0)
      /* Not enough arguments. */
      argp_usage(state);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int
main(int argc, char *argv[])
{
  struct arguments arguments;
  arguments.ntherm = 0;
  arguments.ntraj = 1000;
  arguments.store_every = 1;
  arguments.print_every = 1;
  arguments.delta = 0.1;
  arguments.T = 1;
  arguments.alat = 0.01;
  arguments.mass = 1;
  arguments.omega = 1;
  arguments.target_acc = 0.5;
  arguments.output = "x.txt";
  
  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  
  double delta = arguments.delta;
  double T = arguments.T;
  double alat = arguments.alat;
  double mass = arguments.mass;
  double omega = arguments.omega;
  int ntraj = arguments.ntraj;
  int ntherm = arguments.ntherm;
  double target_acc = arguments.target_acc;
  int store_every = arguments.store_every;
  int print_every = arguments.print_every;
  double N = T/alat;
  double *x = ualloc(sizeof(double)*N);

  /* Rescaled parameters */
  double ma = mass*alat;
  double om = omega*alat;
  
  FILE *fp = uopen(arguments.output, "w");
  normal(N, x, 0, delta);
  for(int i=0; i<ntraj+ntherm; i++) {
    double xt;
    double acc = 0;
    for(int j=0; j<N; j++) {
      normal(1, &xt, x[j], delta);
      double d = dS(N, ma, om, j, xt, x);
      if(d < 0) {
	x[j] = xt;
	acc += 1;
      } else if (urand() < exp(-d)) {
	x[j] = xt;
	acc += 1;
      }
    }
    acc /= N;
    if(i < ntherm) {
      if(acc > target_acc)
	delta *= 1.025;
      if(acc < target_acc)
	delta *= 0.975;
    }
    
    if(i % print_every == 0)
      printf(" i = %8d | a = %6.4f | S(x) = %12.4f | E(x) = %4.2e | racc = %4.2f | delta = %5.3e\n",
	     i, alat, S(N, ma, om, x), E(N, ma, om, x), acc, delta);
    if(i % store_every == 0)
      writex(fp, N, alat, x);
  }
  free(x);
  fclose(fp);
  return 0;
}
  
