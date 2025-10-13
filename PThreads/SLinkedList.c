#include <stdio.h>
#include <stdlib.h>

struct node {
  int data;
  struct node* next;
};

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

int main(int argc, char *argv[]) {
  return 0;
}
