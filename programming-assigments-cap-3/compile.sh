#!/bin/bash

if [ -z "$1" ]; then
  echo "Uso: $0 archivo.c"
  exit 1
fi

file="$1"
name_without_extension="${file%.c}"

echo "Compilando $file con mpicc..."

mpicc -g -Wall -o "$name_without_extension" "$file"

scp $name_without_extension dell:~/

echo "Archivo copiado a dell"

if [ $? -eq 0 ]; then
  echo "Compilación exitosa. Ejecutable: $name_without_extension"
else
  echo "Error en la compilación."
  exit 1
fi
