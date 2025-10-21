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

| Hilos | Tiempo One Mutex for entire list (s) | Tiempo ReadWrite Lock (s) | Aceleración RWLock |
|-------|-------------------|-------------------|--------------------|
| 1     | 0.017105          | 0.016977          | 1.01×              |
| 2     | 0.143237          | 0.061119          | 2.34×              |
| 4     | 0.531598          | 0.152047          | 3.49×              |
| 8     | 4.550090          | 0.727873          | 6.25×              |

Cuando la carga es mayoritariamente de lectura, `rwlock` permite concurrencia entre hilos lectores, lo que resulta en una **mejor escalabilidad a medida que aumenta el número de hilos**, mientras que un único mutex se convierte en un fuerte cuello de botella.

---

##  Resultados — 80 % Member / 10 % Insert / 10 % Delete

| Hilos | Tiempo One Mutex for entire list (s) | Tiempo ReadWrite Lock (s) | Aceleración RWLock |
|-------|-------------------|-------------------|--------------------|
| 1     | 0.519161          | 0.400975          | 1.29×              |
| 2     | 1.635098          | 0.934703          | 1.75×              |
| 4     | 4.229954          | 2.353491          | 1.80×              |
| 8     | 15.328784         | 6.658788          | 2.30×              |

Cuando hay más operaciones de escritura, el `rwlock` **sigue mostrando mejor rendimiento que el mutex global**, aunque las escrituras requieren exclusividad. La mejora es notable frente al mutex porque al menos las operaciones de lectura pueden seguir siendo concurrentes.

---

## Resultados para Matrix por vector usando 4 hilos

| Datos | Tiempo |
|------------|------------|
| 8000000 x 8     |  0.047860    |
| 8000 x 8000     |  0.040008    |
| 8 x 8000000     |  0.096977    |

## Conclusiones

-  Para cargas **dominadas por lecturas**, los *read-write locks* escalan mucho mejor que un mutex global, ya que múltiples hilos pueden leer al mismo tiempo sin bloquearse.  
-  Para cargas **mixtas con escrituras frecuentes**, el `rwlock` mantiene una **ventaja considerable**, aunque reducida a comparación de cuandos se hicieron con 99% para lecturas.  
-  Un mutex único es simple y rápido en baja concurrencia, pero **se degrada drásticamente** al aumentar el número de hilos.  
-  Los *read-write locks* ofrecen **beneficios significativos en escenarios concurrentes intensivos en lectura**, que es precisamente el caso para estructuras de datos compartidas con pocas escrituras.

---
