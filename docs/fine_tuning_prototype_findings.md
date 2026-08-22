# Fine-tuning local en DashAI: implementación exploratoria y hallazgos

## Estado del prototipo

Esta rama implementa un primer flujo completo de fine-tuning eficiente de LLMs dentro de DashAI. El camino validado es:

1. seleccionar un dataset ya registrado en DashAI;
2. mapearlo como `text`, `prompt_completion` o `messages`;
3. ejecutar un preflight de dependencias, hardware, datos y modelo;
4. crear y encolar un `FineTuningRun` persistente con Huey;
5. descargar un modelo base a almacenamiento administrado por DashAI;
6. entrenar un adaptador LoRA o QLoRA y publicar sus artefactos de forma atómica;
7. reiniciar la aplicación;
8. crear una sesión de Generative asociada al run terminado; y
9. cargar el modelo base y el adaptador PEFT para generar una respuesta.

El flujo anterior fue ejecutado de extremo a extremo con QLoRA, Qwen2.5-0.5B-Instruct, 32 ejemplos de Databricks Dolly 15k y una NVIDIA GTX 1080 de 8 GB. Por lo tanto, la integración técnica central es viable. La prueba no constituye todavía una validación de calidad del modelo: se observaron normas de gradiente no finitas y solo se ejecutaron tres pasos.

El prototipo fue consolidado como línea base en el commit `21f093ccb` (rama `memoria/fine-tuning-inferencia`, publicada en el fork personal) y la productización continúa en la rama `memoria/glm-weekend-productization`. Los datos de la prueba están aislados en `E:\DashAI-smoke-finetuning`; no se utilizó ni modificó `E:\DashAI-data`.

## Arquitectura implementada

### Persistencia y almacenamiento

Se incorporó la entidad `FineTuningRun`, que conserva el dataset, modelo y revisión, método, mapeo, hiperparámetros, estado, identificador Huey, progreso, métricas, entorno, artefacto, error, cancelación y fechas. Los estados posibles son `not_started`, `queued`, `running`, `completed`, `failed` y `canceled`.

`GenerativeSession` posee ahora una referencia nullable a un run. La migración mantiene compatible el camino antiguo: una sesión sin `fine_tuning_run_id` continúa resolviendo su modelo únicamente por `model_name`.

Se agregaron dos raíces configurables:

- `LLM_MODELS_PATH`: snapshots de modelos base curados;
- `FINE_TUNING_PATH`: adaptadores, tokenizer, plantilla de chat, estado y manifiestos.

Las operaciones destructivas validan que la ruta esté dentro del almacenamiento administrado por DashAI. No se permite borrar un run activo, un adaptador usado por una sesión ni una ruta externa. La publicación del resultado usa un directorio temporal y un renombrado final, de modo que un run no aparece como completado con un artefacto parcial.

### Motor de entrenamiento

`FineTuningBackend` separa el contrato de entrenamiento de Huey, SQLAlchemy y la API. La implementación de referencia usa Transformers, Datasets, TRL `SFTTrainer`, PEFT y bitsandbytes. Esto permite reemplazar o complementar el motor con Unsloth sin cambiar la persistencia ni la interfaz HTTP.

El catálogo inicial se limita deliberadamente a Qwen2.5-0.5B-Instruct y Qwen2.5-1.5B-Instruct. `snapshot_download` guarda una revisión resuelta en `.dashai_model.json`; el manifiesto de cada adaptador registra tanto la revisión solicitada como el hash resuelto.

Los formatos admitidos son:

- `text`: una columna con el texto de entrenamiento;
- `prompt_completion`: columnas separadas de instrucción y respuesta;
- `messages`: lista o JSON serializado con roles `system`, `user` y `assistant`.

La preparación rechaza columnas ausentes, ejemplos vacíos, JSON inválido y roles desconocidos. Además aplica límite de muestras, división de validación reproducible, preview y fingerprint sobre los datos ya mapeados.

QLoRA carga el modelo en NF4 con doble cuantización y cómputo FP16, y PEFT usa `target_modules="all-linear"`. LoRA usa precisión FP16 en CUDA para estos modelos pequeños. El preset `quick_test` desactiva evaluación, checkpoints intermedios y gradient checkpointing para que tres pasos sean una prueba de integración razonable en Pascal; los presets de trabajo conservan configuraciones más representativas.

### Jobs, API e inferencia

`FineTuningJob` actualiza estado y progreso, guarda métricas y traduce errores a un estado persistente. La cancelación elimina de Huey un trabajo todavía en cola o solicita detención cooperativa si ya está entrenando. El callback consulta la solicitud al final de cada paso, por lo que no equivale a una interrupción inmediata del proceso.

La API implementada bajo `/api/v1/fine-tuning` expone catálogo, preflight, creación/listado/detalle de runs, inicio, cancelación, borrado e inventario. El inventario combina modelos base y adaptadores y reporta procedencia, tamaño, estado, run y ruta.

`PeftAdapterTextGenerationModel` es un componente generativo interno. `GenerativeJob` inyecta las rutas validadas del modelo base y del adaptador cuando una sesión referencia un run. En GPU se intenta cargar el modelo base cuantizado a cuatro bits; en CPU se usa precisión completa. El modelo interno se oculta del asistente normal y del selector de modelos, porque solo es válido cuando la sesión fue creada desde un run completado. Los modelos GGUF existentes no fueron modificados.

### Interfaz

La ruta `/app/generative/fine-tuning` contiene:

- lista y detalle de ejecuciones;
- Stepper de modelo/hardware, dataset, configuración y revisión;
- parámetros avanzados construidos mediante `FormSchemaLayout`;
- preflight, preview, advertencias y bloqueos;
- progreso, métricas, error, cancelación y eliminación;
- acción **Open in Generative**; e
- inventario con borrado controlado.

La página tiene textos mínimos en español e inglés y usa inglés como fallback. Se añadió el acceso desde el inicio de Generative sin crear un quinto módulo principal.

## Reproducción y verificación

Desde la raíz del repositorio, las dependencias opcionales declaradas son:

```powershell
uv sync --extra finetuning
```

Para esta prueba se conservaron explícitamente los binarios CUDA existentes de PyTorch 2.11.0 y se instalaron solo las dependencias nuevas:

```powershell
uv pip install --python .venv\Scripts\python.exe trl==1.10.0 peft==0.20.0 bitsandbytes==0.50.1
```

Pruebas backend relevantes:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/back/fine_tuning tests/back/api/test_fine_tuning_api.py
```

Pruebas y build frontend:

```powershell
Set-Location DashAI\front
yarn test --watchAll=false --runTestsByPath src/api/fineTuning.test.ts src/pages/generative/FineTuning.test.jsx
yarn build
```

Prueba real aislada:

```powershell
.venv\Scripts\python.exe scripts/fine_tuning_smoke.py --local-path E:\DashAI-smoke-finetuning
```

El script importa una muestra reproducible de Dolly, usa las API de DashAI, ejecuta Huey en modo inmediato, reinicia la aplicación y genera una respuesta mediante `GenerativeJob`. El resultado estructurado queda en `E:\DashAI-smoke-finetuning\smoke_result.json`.

La verificación final obtuvo 33 pruebas backend y 5 pruebas frontend exitosas, además de compilación Python, Ruff sobre los archivos afectados, migración upgrade/downgrade y build frontend de producción. El lint global del frontend no es actualmente una señal utilizable: la configuración del repositorio intenta analizar TypeScript sin el parser correspondiente y tampoco carga `react-hooks`, por lo que reporta errores equivalentes en decenas de archivos existentes. Este problema de configuración no se corrigió porque está fuera del alcance del prototipo; las pruebas y el compilador sí procesaron los archivos nuevos correctamente.

## Resultado de la prueba real

| Elemento | Resultado |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1080, 8 GB, compute capability 6.1 |
| Entorno | Windows 10, Python 3.11.2, CUDA 12.6 |
| PyTorch | 2.11.0+cu126 |
| Transformers / TRL / PEFT / bitsandbytes | 4.57.6 / 1.10.0 / 0.20.0 / 0.50.1 |
| Modelo | Qwen2.5-0.5B-Instruct |
| Revisión resuelta | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Dataset | Dolly 15k, muestra aleatoria de 32 filas con seed 42 |
| División | 28 entrenamiento / 4 validación |
| Fingerprint | `48bea54d74e77b2e5d70518fc412af2c6ea78fe3efaa7bf0138a9d4a154816a4` |
| Configuración | QLoRA NF4, rank 8, longitud 128, batch 1, tres pasos |
| Tiempo de entrenamiento | 497.97 s |
| Tiempo del backend | 504.83 s |
| Tiempo total dataset → reinicio → inferencia | 519.27 s |
| Rendimiento | 0.006 pasos/s, aproximadamente 166 s por paso |
| Pérdida por paso | 5.0801; 2.6988; 3.9637 |
| Pérdida final informada | 3.9142 |
| Pico de VRAM medido por PyTorch | 1,610,979,840 bytes, aproximadamente 1.50 GiB |
| Tamaño de `adapter/` | 33,525,868 bytes |
| Tamaño total del artefacto | 33,533,302 bytes |
| Tamaño del modelo base local | 999,605,379 bytes |
| Inferencia después de reiniciar | exitosa, 8.70 s |

La respuesta generada fue: “One primary color is red, which is a bright, energetic, and eye-catching color that can evoke feelings of passion, energy, and excitement. In one sentence”. Esto comprueba la carga del snapshot local y del adaptador persistido, pero no permite atribuir una mejora al fine-tuning.

## Problemas encontrados y decisiones tomadas

### Gradientes BF16 incompatibles con Pascal

El primer intento falló en el primer backward con `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`. Aunque el cómputo cuantizado se había solicitado en FP16, algunos parámetros entrenables quedaron en BF16. La solución aplicada fuerza a FP32 todos los parámetros entrenables después de construir `SFTTrainer`; el manifiesto del run exitoso registra 336 parámetros entrenables en FP32.

### Gradient checkpointing en una prueba de tres pasos

`prepare_model_for_kbit_training` activa gradient checkpointing por defecto. En esta GPU elevó mucho el costo de una prueba cuyo objetivo era verificar integración, no capacidad de secuencia larga. `quick_test` lo desactiva; `qlora_8gb` lo conserva para su objetivo de memoria.

### Estabilidad numérica pendiente

Los tres valores de pérdida fueron finitos y el artefacto pudo usarse en inferencia, pero `grad_norm` fue `NaN` en todos los pasos. Por eso no debe declararse que QLoRA es numéricamente estable en esta combinación de Pascal, PyTorch, bitsandbytes y versiones de entrenamiento. Se requiere una prueba más larga comparando escalado de pérdida, learning rate, optimizador, dtype de capas normalizadoras y versiones compatibles. También conviene convertir esta señal en un criterio de fallo o advertencia configurable.

> Actualización (consolidación): la prueba real de 15 pasos se ejecutó y la señal quedó instrumentada; ver «Prueba real de estabilidad» más abajo.

### Cancelación y recuperación

La cancelación cooperativa se observa en el límite de un paso. En la GTX 1080 un paso tomó cerca de tres minutos, por lo que la latencia puede ser demasiado alta para la experiencia esperada. Una cancelación fuerte requeriría aislar cada entrenamiento en un subproceso administrado. La reconciliación de runs huérfanos al arrancar fue implementada en la fase de consolidación (ver «Consolidación y robustez»); el script de smoke también limpia sus propios intentos anteriores.

> Actualización posterior: la cancelación fuerte fue implementada mediante aislamiento en proceso hijo; ver «Cancelación fuerte por aislamiento en proceso» más abajo.

## Consolidación y robustez (rama `memoria/glm-weekend-productization`)

La línea base del prototipo se robusteció en cinco frentes. Todo lo listado aquí está cubierto por pruebas automatizadas; lo que requiere hardware se marca explícitamente.

### Recuperación al arrancar

`DashAI/back/fine_tuning/reconciliation.py` se ejecuta dentro de `create_app` después de las migraciones. Es idempotente y:

- marca `failed` todo run `running` con el mensaje «Training was interrupted because DashAI stopped...» (nunca lo completa por la presencia de un directorio);
- conserva un run `queued` solo si su tarea sigue pendiente en la tabla `task` de la cola Huey (nuevo método `HueyJobQueue.task_snapshot`); si la tarea desapareció, lo marca `failed`. Si un worker vivo estaba ejecutando la tarea, el propio job la re-marca `running`/`completed`, de modo que el estado final siempre es el veraz;
- pone en cuarentena (renombra a `orphaned-<nombre>-<fecha>`, no borra) los directorios `*.tmp` directamente dentro de `FINE_TUNING_PATH`;
- elimina cualquier lock GPU residual.

Un run interrumpido puede reiniciarse desde la interfaz porque `start` solo rechaza estados activos o `completed`.

### Exclusión de GPU entre procesos

`DashAI/back/fine_tuning/resource_lock.py` implementa un lock basado en archivo con creación atómica (`O_CREAT|O_EXCL`) dentro de `FINE_TUNING_PATH/gpu.lock`. El archivo registra pid y dueño; un lock cuyo proceso dueño murió se descarta, y `release` se niega a liberar un lock ajeno. La comprobación ocurre en dos niveles: `POST /runs/{id}/start` responde 409 con mensaje accionable, y el worker re-verifica dentro de `FineTuningJob.run` (espera acotada de 10 s) para cerrar la carrera entre dos inicios casi simultáneos. La liberación ocurre en un `finally` que cubre éxito, cancelación y excepción. La inferencia con adaptador (`GenerativeJob`) detecta el lock antes de cargar el modelo y falla con el mismo mensaje accionable en lugar de arriesgar un OOM.

### Liberación de recursos

`FineTuningJob.run` libera el lock en `finally`. El backend de Transformers suelta tokenizer, modelo, trainer y datasets con `del` dentro de su propio `finally` antes de `gc.collect()` y `torch.cuda.empty_cache()`: el `del` rebinda las variables del frame aunque un traceback conserve el frame durante la propagación del error, por lo que la VRAM queda disponible antes de que la excepción llegue a Huey.

### Salud numérica

El callback de entrenamiento registra por paso: pérdida, norma de gradiente, learning rate, paso, época (vía `logs` de HF), duración de cada paso y advertencias. Al terminar, `metrics` incluye `health_warnings` y `last_logged`, y `runtime_metadata` agrega `health` con duraciones por paso y su media, pico de VRAM y dtypes entrenables. Reglas de comportamiento:

- pérdida no finita → el run falla (`TrainingHealthError`) y el adaptador no se publica;
- `grad_norm` no finito → advertencia estructurada visible como Alert en la página de runs;
- `_json_safe` convierte NaN/Inf a `null` para que el JSON de la API sea siempre válido.

### Internacionalización

`FineTuning.jsx` abandonó su diccionario manual inglés/español: ahora usa `useTranslation(["generative", "common"])` con claves nuevas bajo `generative:fineTuning.*` (traducciones completas en `en` y `es`; `pt`, `de` y `zh` caen al fallback estándar inglés). Los textos hardcodeados del wizard y de `CreateSessionLanding.jsx` fueron migrados, las advertencias numéricas se muestran como Alert, y los parámetros del paso de configuración se separaron en básicos y avanzados (plegable).

### Validación de la consolidación

- Backend: `python -m pytest -q tests/back/fine_tuning tests/back/api/test_fine_tuning_api.py tests/back/api/test_fine_tuning_robustness.py` — 40 pruebas pasando (reconciliación y su idempotencia, cuarentena de temporales, lock y su toma de control sobre pids muertos, exclusión y liberación en éxito/error, sanidad de JSON, mensajes de GPU ocupada).
- Frontend: `yarn test --runTestsByPath src/api/fineTuning.test.ts src/pages/generative/FineTuning.test.jsx` — 6 pruebas, incluida una que cambia el idioma a español y verifica claves i18n reales.
- Ruff sin errores en los archivos modificados; build de producción del frontend verificado.
- La política de pérdida no finita y las advertencias de `grad_norm` están validadas con pruebas unitarias sobre los helpers; su comportamiento en GPU real se evalúa en la prueba de estabilidad descrita más abajo.

#### Prueba real de estabilidad (15 pasos, QLoRA, GTX 1080)

`scripts/fine_tuning_stability.py` ejecutó 15 pasos con 64 ejemplos de Dolly (seed 42) contra `E:\DashAI-smoke-finetuning`, reutilizando el modelo ya descargado. Resultado en `stability_result.json`: estado `completed`, 2081 s en total, 136,6 s/paso de media, pérdida finita y decreciente (último valor registrado 1,58; `train_loss` final 2,74), pico de VRAM 1,70 GB, 336 parámetros entrenables en FP32, learning rate con calendario lineal visible en `last_logged`.

**Hallazgo principal**: `grad_norm` fue `NaN` solo en los pasos 1–4; del paso 5 al 15 fue finito (3,02 en el último). El NaN del smoke de 3 pasos es, por tanto, transitorio y autolimitado en esta configuración, no una inestabilidad persistente. La instrumentación nueva lo reporta exactamente como cuatro advertencias estructuradas sin fallar el run (la pérdida siempre fue finita), que es el comportamiento especificado. No se ejecutaron las dos configuraciones adicionales permitidas: la evidencia caracteriza el fenómeno y el presupuesto de GPU se reserva para el resto de la sesión. Tiempo de GPU consumido hasta ahora: ~35 minutos.

## Cancelación fuerte por aislamiento en proceso

Los backends estándar (Transformers y Unsloth) ahora entrenan en un **proceso hijo gestionado**:

- `DashAI/back/fine_tuning/training_worker.py` es el hijo: se lanza con `python -m DashAI.back.fine_tuning.training_worker <run_id>`, construye una configuración ligera (sin registro de componentes, evitando importar torch/diffusers/timm al arrancar) y ejecuta el run completo (validación de dataset, `ensure_model`, entrenamiento) reportando progreso, métricas y estado final **solo por la base de datos**.
- `DashAI/back/fine_tuning/training_process.py` es el lado del padre dentro de `FineTuningJob`: lanza el hijo, reenvía el progreso de la BD a la cola de jobs, y aplica la política de cancelación en tres niveles: (1) el hijo observa la bandera cooperativa entre pasos; (2) al solicitar cancelación tiene una gracia de 30 s para terminar solo; (3) si no responde, se emite `terminate()` y, tras 10 s, `kill()`. La muerte del proceso destruye su contexto CUDA y libera la VRAM al instante, que es lo que hace fuerte a la cancelación en Windows. `wait()` garantiza que nunca queda un proceso huérfano.
- El **lock GPU permanece en el padre** (el worker de Huey), que lo libera en `finally` para éxito, cancelación, fallo y kill. Si el worker muriera bruscamente, la reconciliación del siguiente arranque limpia el lock.
- Los backends inyectados por el contenedor (pruebas, futuros plugins) conservan la ruta en proceso (`ISOLATED_TRAINING_ENABLED`, desactivable con `DASHAI_ISOLATED_TRAINING=0`).
- Si el hijo muere sin dejar estado terminal, el padre marca `failed` con el mensaje del hijo o con el código de salida; si la muerte ocurrió tras una solicitud de cancelación, marca `canceled`.

### Validación de la cancelación fuerte

- Pruebas reales de subproceso en Windows (4): hijo que completa (artefacto y métricas persistidos por el hijo), cancelación cooperativa dentro de la gracia (hilo externo marca la bandera cuando el hijo reporta progreso), terminación forzosa de un hijo que ignora la bandera (el retorno solo ocurre tras `wait()`: sin procesos huérfanos) y fallo del hijo reportado con su mensaje. Además, una integración a nivel de job valida la ruta completa por API con adquisición y liberación del lock.
- **Ejecución real en GPU**: un run QLoRA de 3 pasos (run 5, `E:\DashAI-smoke-finetuning`, `stability_result.json`) entrenó íntegramente por la ruta aislada: el padre mantuvo `gpu.lock` y el hijo entrenó (pasos de 115–137 s, pico 1,66 GB, advertencias NaN en los 3 pasos, coherente con el patrón temprano documentado) y persistió progreso, métricas y estado final por la base de datos.
- Batería focalizada completa: 65 pruebas backend pasando.

## Administración e inferencia local (Hito 2)

MVP inspirado en LM Studio, limitado al flujo base+adaptadores de Transformers:

- **Entidad `ManagedLocalModel`** (migración `c3f5a9b7d2e1`, verificada con upgrade/downgrade/upgrade sobre base temporal) que trackea descargas con estados `downloading`/`ready`/`error`, revisión resuelta y tamaño.
- **Descarga mediante jobs**: `ManagedModelDownloadJob` reutiliza `ensure_model` (mismo almacenamiento administrado y publicación atómica) y reporta progreso por el sistema de jobs existente. `POST /api/v1/fine-tuning/models/{key}/download` responde 202 y rechaza duplicados con 409.
- **Inventario unificado** en `GET /models`: modelos del catálogo no descargados (con VRAM recomendada y botón de descarga), descargas en curso/errores con reintentos, snapshots adoptados (los descargados por preflight antes de esta entidad se reportan `ready` y la descarga los adopta sin redescargar) y adaptadores con su run. La eliminación se bloquea (409) si el modelo está descargando, en un run activo o referenciado por una sesión.
- **Componente interno `LocalManagedTextGenerationModel`** (oculto del selector como `PeftAdapterTextGenerationModel`): resuelve el modelo por `local_model_id` del inventario — nunca por una ruta enviada por el cliente —, aplica la plantilla de chat, usa `trust_remote_code=False`, soporta `device` CPU/GPU y expone `top_k` y `seed` (también añadidos, opcionales, al adaptador PEFT para compatibilidad con sesiones antiguas).
- **`GenerativeSession.local_model_id`** nullable con FK `SET NULL`; las sesiones históricas sin la referencia siguen funcionando. `GenerativeJob` valida estado `ready`, resuelve la ruta administrada y comprueba el lock GPU antes de cargar.
- **Frontend**: el inventario muestra estado con chip traducido, VRAM recomendada, descarga/reintento, apertura de sesión desde modelo base (`LocalManagedTextGenerationModel`) y desde adaptador; polling mientras hay descargas activas.

### Validación del Hito 2

- Ejecución real (`scripts/local_model_inference_check.py`, resultado en `base_inference_result.json`): adopción del snapshot existente vía endpoint de descarga, sesión desde el modelo base con `top_k`/`seed`, y generación correcta («One primary color is red...») en 19,2 s sobre la GTX 1080. Con esto el MVP cumple los seis criterios de aceptación: descargar/reutilizar modelo curado, inventario, sesión desde base, parámetros de generación, respuesta generada, y el flujo equivalente con adaptador ya validado en el smoke del prototipo.
- Pruebas automatizadas (51 backend + 8 frontend pasando): inventario con modelos no descargados, descarga con estado y 409 en duplicados, eliminación bloqueada por sesión, validación de referencias `local_model_id` (404/400), inyección de `_base_model_path` en `GenerativeJob` mediante registro simulado, descarga y sesión desde base en la interfaz, y compatibilidad de sesiones antiguas.

#### Comparación real base vs adaptador

`scripts/base_vs_adapter_report.py` generó con parámetros idénticos (greedy, seed 42, `top_k` 50, 48 tokens) sobre el modelo base administrado y sobre el adaptador del run de 15 pasos (resultado en `base_vs_adapter_result.json`). Las tres respuestas son coherentes en ambos casos (4–10 s por generación), y el adaptador muestra una **deriva de estilo reproducible** hacia el formato instrucción-respuesta de Dolly (p. ej. abre con «In one sentence, I could say: ...»), evidencia cualitativa de que el ajuste afectó el comportamiento sin degradar la fluidez. No es un benchmark de calidad: para eso se requiere un protocolo con métricas y conjunto de evaluación.

## Backend Unsloth opcional (Hito 3)

Unsloth se modeló como **backend de ejecución** independiente del método LoRA/QLoRA:

- `FineTuningBackendType` (`transformers`/`unsloth`) y columna `FineTuningRun.backend` con valor heredado `transformers` (migración `d4a7c1e9f3b2`, verificada con upgrade sobre una fila existente — el default aplica —, downgrade y re-upgrade).
- Selección de backend en schemas, API (`POST /runs` acepta y persiste `backend`), preflight y wizard (el selector Unsloth solo aparece cuando el catálogo lo reporta disponible).
- `UnslothFineTuningBackend` implementa el contrato con imports completamente diferidos (la importación de Unsloth parchea el estado global de torch y ocurre solo en el worker que entrena), produce el mismo layout de artefacto (adaptador, manifiesto con `backend`, trainer_state, plantilla de chat) y reutiliza los chequeos de salud numérica del backend Transformers.
- Capacidades dinámicas en el catálogo vía `find_spec` (sin importar Unsloth ni torch: verificado que `get_catalog()` corre sin cargar torch) con `unsloth`, `unsloth_version` y `unsloth_reason`.
- Preflight: backend `unsloth` sin instalar produce el blocker `unsloth_not_installed`; instalado sobre una GPU con compute capability < 7.0 produce la advertencia `unsloth_experimental_gpu` (Pascal tratado como experimental, no bloqueado).

### Veredicto de compatibilidad (investigación aislada)

En un venv temporal (`E:/tmp-unsloth-venv`, Python 3.11.2) se resolvió `pip install unsloth --dry-run`: `unsloth 2026.8.19` requeriría `torch 2.11.0` (rueda pypi, que en Windows empaqueta CUDA 12.8 — la combinación que este equipo descartó por no admitir la GTX 1080), además de `transformers 5.5.0`, `trl 0.24.0`, `torchvision 0.26.0`, `xformers`, `triton-windows` y `torchao`, reemplazando el stack fijado de DashAI (`transformers 4.57.6`, `trl 1.10.0`, `peft 0.20.0`, `bitsandbytes 0.50.1`) y la instalación funcional de PyTorch `2.11.0+cu126`. Por ese motivo:

- **no** se añadió el extra `finetuning-unsloth` a `pyproject.toml`: no existe una combinación reproducible que no sustituya las dependencias base;
- no se instaló Unsloth en la `.venv` principal ni se ejecutó el smoke de 3 pasos;
- el backend queda implementado, detectado y **deshabilitado honestamente** en este equipo (`capabilities.unsloth = false` con motivo accionable).

Condiciones para habilitarlo después: una GPU con compute capability ≥ 7.0 y una ventana para re-pin del stack (o un entorno/worker aislado con su propio intérprete), junto con la validación del contrato de `FastLanguageModel` de la versión que se resuelva en ese momento.

### Validación del Hito 3

- 60 pruebas backend pasando, incluidas: catálogo con y sin Unsloth, default `transformers` en creación, selección del backend Unsloth en el worker mediante sustitución simulada (el run completa con `metrics.backend = "unsloth"`), preflight con blocker/advertencia según instalación y GPU, y mensaje claro al entrenar sin Unsloth.
- Migración verificada con upgrade sobre dato heredado, downgrade y re-upgrade; `get_catalog()` confirmado sin cargar torch.
- El backend Unsloth queda implementado pero no ejecutado en GPU real por las incompatibilidades documentadas arriba.

### Semántica del downgrade

La migración fue verificada con upgrade, downgrade al head anterior y un nuevo upgrade. Como es habitual para una tabla creada completamente por una migración, el downgrade elimina los registros de fine-tuning; los archivos de adaptadores permanecen en disco y pasan a ser huérfanos. Antes de aceptar esta migración se debe decidir si el downgrade destructivo es suficiente, si se requiere una advertencia explícita o si DashAI necesita una herramienta de respaldo y reimportación de manifiestos.

### Alcance deliberadamente limitado

No se implementaron Unsloth, GGUF, publicación a Hugging Face Hub, entrenamiento distribuido, checkpoints reanudables, búsqueda remota ni modelos arbitrarios. El inventario es únicamente local. Estas exclusiones mantienen el prototipo centrado en la brecha principal: dataset DashAI → ajuste persistente → inferencia Generative.

## Preguntas para la reunión con Felipe

1. ¿Esta capacidad debe vivir en el núcleo o como uno o más plugins? ¿Qué partes se consideran API estable?
2. ¿Se planea recuperar el diseño de trabajos basado en unidades atómicas que existió y fue revertido, o `FineTuningJob` debe adaptarse al sistema actual?
3. ¿Windows y GPU Pascal deben tener soporte oficial, soporte de mejor esfuerzo o quedar fuera del mínimo? ¿Cuál será el piso de VRAM?
4. ¿Unsloth es un requisito del tema, un acelerador opcional o solo una alternativa que se debe estudiar?
5. ¿Qué familias, tamaños y licencias de modelos son prioritarios? ¿Cómo debe operar DashAI con repositorios restringidos?
6. ¿Qué formatos de dataset deben considerarse parte del MVP y cuánto del módulo Datasets se debe reutilizar o extender?
7. ¿El producto final debe conservar solo el adaptador PEFT o también producir modelo fusionado, GGUF o publicación en Hub?
8. ¿Qué profundidad se espera del inventario estilo LM Studio: solo almacenamiento local, descarga remota, cuantizaciones, servidor OpenAI-compatible o administración completa?
9. ¿Cancelación fuerte, checkpoints y reanudación son requisitos de aceptación? Si lo son, ¿es aceptable aislar entrenamientos en subprocesos?
10. ¿Cómo se deben reconciliar runs huérfanos después de cerrar DashAI y cómo se planifica la concurrencia cuando varios trabajos compiten por VRAM?
11. ¿Qué pesa más en la evaluación: calidad del modelo, consumo de recursos, reproducibilidad o usabilidad?
12. ¿Habrá hardware adicional, datasets de referencia y participantes disponibles para las pruebas de usabilidad?
13. ¿Cuál es la política del proyecto para extras opcionales pesados, migraciones y compatibilidad hacia atrás?
14. ¿Una norma de gradiente no finita debe abortar el run, generar una advertencia o evaluarse junto con pérdida y métricas posteriores?
15. ¿El modelo PEFT genérico debe permanecer como componente interno o se desea que usuarios avanzados puedan configurarlo manualmente?
16. ¿El preset rápido debe omitir evaluación y checkpoints, o todo run debe producir métricas sobre validación aunque sea una prueba de humo?

## Siguientes pasos técnicos recomendados

Estado tras la productización: la reconciliación al arrancar, la programación exclusiva de GPU, la instrumentación de estabilidad (con la prueba real de 15 pasos), la internacionalización completa, el inventario local con descargas y sesión desde modelo base, la cancelación fuerte por aislamiento en proceso y una primera comparación cualitativa base vs adaptador ya están implementados y validados. Lo pendiente, en orden de impacto: un protocolo de evaluación de calidad base vs adaptado con métricas y conjunto de evaluación; una prueba de estabilidad más larga (40–50 pasos) y con warmup explícito para caracterizar el NaN temprano de `grad_norm` (añadir `warmup_steps` a `TrainingParameters`); latencia de cancelación observada en UI (hoy: gracia cooperativa de 30 s + kill); y la decisión con el profesor sobre plugins, artefactos (fusión/GGUF) y modelos soportados antes de ampliar el catálogo. Unsloth queda habilitable cuando exista hardware con compute capability ≥ 7.0 o una ventana para re-pin del stack.
