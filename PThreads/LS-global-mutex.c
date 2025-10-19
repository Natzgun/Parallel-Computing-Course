#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define TOTAL_OPS 100000

struct node {
  int data;
  struct node* next;
};

struct node* head = NULL;
pthread_mutex_t mutex;
int nthreads = 1;

double member_frac = 0.99;
double insert_frac = 0.005;
double delete_frac = 0.005;

void Push(struct node **headRef, int data) {
  struct node *newNode = malloc(sizeof(struct node));
  newNode->data = data;
  newNode->next = *headRef;
  *headRef = newNode;
}

int Member(int value, struct node* curr_head_p) {
  while (curr_head_p) {
    if (curr_head_p->data < value)
      curr_head_p = curr_head_p->next;
    else
      break;
  }
  if (!curr_head_p || curr_head_p->data > value)
    return 0;
  else
    return 1;
}

int Insert(struct node **head, int value) {
  struct node *curr_p = *head;
  struct node* pred_p = NULL;

  while (curr_p && curr_p->data < value) {
    pred_p = curr_p;
    curr_p = curr_p->next;
  }

  if (curr_p && curr_p->data == value)
    return 0;

  if (!pred_p)
    Push(head, value);
  else
    Push(&(pred_p->next), value);

  return 1;
}

int Delete(struct node **head, int value) {
  struct node *curr_p = *head;
  struct node *pred_p = NULL;

  while (curr_p && curr_p->data < value) {
    pred_p = curr_p;
    curr_p = curr_p->next;
  }

  if (!curr_p || curr_p->data > value)
    return 0;

  if (pred_p == NULL) {
    *head = curr_p->next;
  } else {
    pred_p->next = curr_p->next;
  }

  free(curr_p);
  return 1;
}

int ThreadSafeMember(int value) {
  pthread_mutex_lock(&mutex);
  int result = Member(value, head);
  pthread_mutex_unlock(&mutex);
  return result;
}

int ThreadSafeInsert(int value) {
  pthread_mutex_lock(&mutex);
  int result = Insert(&head, value);
  pthread_mutex_unlock(&mutex);
  return result;
}

int ThreadSafeDelete(int value) {
  pthread_mutex_lock(&mutex);
  int result = Delete(&head, value);
  pthread_mutex_unlock(&mutex);
  return result;
}

void* ThreadFunc(void* rank) {
  long my_rank = (long) rank;
  int ops_per_thread = TOTAL_OPS;

  for (int i = 0; i < ops_per_thread; i++) {
    int val = rand() % 10000;
    double prob = (double) rand() / RAND_MAX;

    if (prob < member_frac) {
      ThreadSafeMember(val);
    } else if (prob < member_frac + insert_frac) {
      ThreadSafeInsert(val);
    } else {
      ThreadSafeDelete(val);
    }
  }
  return NULL;
}

void PrintList(struct node* head) {
  struct node* curr = head;
  printf("Lista actual: ");
  while (curr) {
    printf("%d -> ", curr->data);
    curr = curr->next;
  }
  printf("NULL\n");
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Uso: %s <num_threads> [member_frac insert_frac delete_frac]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  nthreads = atoi(argv[1]);
  if (argc == 5) {
    member_frac = atof(argv[2]);
    insert_frac = atof(argv[3]);
    delete_frac = atof(argv[4]);
  }

  srand((unsigned int) time(NULL));
  pthread_t threads[nthreads];
  pthread_mutex_init(&mutex, NULL);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (long i = 0; i < nthreads; i++) {
    pthread_create(&threads[i], NULL, ThreadFunc, (void*) i);
  }

  for (int i = 0; i < nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nTiempo total: %.6f segundos\n", elapsed);
  printf("Threads: %d | Operaciones: %d | Distribución: %.0f%% Member, %.0f%% Insert, %.0f%% Delete\n",
         nthreads, TOTAL_OPS,
         member_frac * 100, insert_frac * 100, delete_frac * 100);

  pthread_mutex_destroy(&mutex);
  return 0;
}
