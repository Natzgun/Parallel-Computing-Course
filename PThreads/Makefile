CC = gcc
CFLAGS = -g -Wall
LIBS = -lpthread

%: %.c
	$(CC) $(CFLAGS) -o $*.out $< $(LIBS)

clean:
	rm -f *.out
