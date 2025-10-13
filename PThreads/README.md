# Linked List Concurrent Access Performance

##  Configuración de Pruebas

- **Plataforma:** Arch Linux (POSIX Threads)  
- **Número de operaciones por hilo:** `100000`  
- **Medida de tiempo:** `clock_gettime` con `CLOCK_MONOTONIC`  
- **Escenarios probados:**
  - 99.9 % Member — 0.05 % Insert — 0.05 % Delete *(casi solo lectura)*  
  - 80 % Member — 10 % Insert — 10 % Delete *(más operaciones de escritura)*

---
## Resultados del Libro
<img width="488" height="451" alt="image" src="https://github.com/user-attachments/assets/4acdcb69-f30d-4c9d-a9c1-988c10af00ad" />


---
## Resultados — 99.9 % Member

| Hilos | Tiempo One Mutex for entire list | Tiempo ReadWrite Lock (s) | Aceleración RWLock |
|-------|-------------------|-------------------|--------------------|
| 1     | 0.017105          | 0.016977          | 1.01×              |
| 2     | 0.060067          | 0.061119          | 0.98×              |
| 4     | 0.150434          | 0.152047          | 0.99×              |
| 8     | 0.760817          | 0.727873          | 1.05×              |


Cuando la carga es mayoritariamente de lectura, `rwlock` permite concurrencia entre hilos lectores, logrando un rendimiento similar o ligeramente superior al mutex tradicional en cargas altas (8 hilos).

---

##  Resultados — 80 % Member / 10 % Insert / 10 % Delete

| Hilos | Tiempo One Mutex for entire list | Tiempo ReadWrite Lock (s) | Aceleración RWLock |
|-------|-------------------|-------------------|--------------------|
| 1     | 0.384180          | 0.400975          | 0.96×              |
| 2     | 0.938453          | 0.934703          | 1.00×              |
| 4     | 2.264606          | 2.353491          | 0.96×              |
| 8     | 6.430782          | 6.658788          | 0.97×              |

Cuando hay más operaciones de escritura, el `rwlock` **pierde ventaja** porque los escritores requieren acceso exclusivo. La diferencia entre ambas técnicas se vuelve marginal o incluso negativa.

---


## Conclusiones

-  Para cargas **dominadas por lecturas**, los *read-write locks* permiten una mejora leve al permitir acceso concurrente de múltiples hilos lectores.  
-  Para cargas **mixtas con escrituras frecuentes**, la ganancia desaparece ya que las escrituras requieren exclusividad.  
-  Un mutex único tiene un costo bajo y es más simple de implementar, mientras que `rwlock` ofrece beneficios **solo en escenarios de lectura intensiva** como lo menciona en el libro.
---
