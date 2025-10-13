#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

struct node {
  int data;
  struct node* next;
};

struct node* head = NULL;
pthread_rwlock_t rwlock;

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
  pthread_rwlock_rdlock(&rwlock);
  int result = Member(value, head);
  pthread_rwlock_unlock(&rwlock);
  return result;
}

int ThreadSafeInsert(int value) {
  pthread_rwlock_wrlock(&rwlock);
  int result = Insert(&head, value);
  pthread_rwlock_unlock(&rwlock);
  return result;
}

int ThreadSafeDelete(int value) {
  pthread_rwlock_wrlock(&rwlock);
  int result = Delete(&head, value);
  pthread_rwlock_unlock(&rwlock);
  return result;
}

void* ThreadFunc(void* rank) {
  long my_rank = (long) rank;

  ThreadSafeInsert(10 + my_rank);
  ThreadSafeInsert(20 + my_rank);
  ThreadSafeInsert(30 + my_rank);

  int found = ThreadSafeMember(20 + my_rank);
  printf("Thread %ld: Member(20+rank) = %d\n", my_rank, found);

  ThreadSafeDelete(10 + my_rank);

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
  pthread_t threads[4];
  pthread_rwlock_init(&rwlock, NULL);

  for (long i = 0; i < 4; i++) {
    pthread_create(&threads[i], NULL, ThreadFunc, (void*) i);
  }

  for (int i = 0; i < 4; i++) {
    pthread_join(threads[i], NULL);
  }

  pthread_rwlock_rdlock(&rwlock);
  PrintList(head);
  pthread_rwlock_unlock(&rwlock);

  pthread_rwlock_destroy(&rwlock);
  return 0;
}
