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

Los cambios se mantienen sin commit y sin push. Los datos de la prueba están aislados en `E:\DashAI-smoke-finetuning`; no se utilizó ni modificó `E:\DashAI-data`.

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

### Cancelación y recuperación

La cancelación cooperativa se observa en el límite de un paso. En la GTX 1080 un paso tomó cerca de tres minutos, por lo que la latencia puede ser demasiado alta para la experiencia esperada. Una cancelación fuerte requeriría aislar cada entrenamiento en un subproceso administrado. Además, la aplicación todavía no reconcilia automáticamente runs `running` que quedaron huérfanos por cierre abrupto; el script de smoke sí limpia sus propios intentos anteriores, pero eso no reemplaza una política de producción.

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

Antes de convertir este prototipo en una entrega de producto, conviene priorizar: una prueba de estabilidad de 20–50 pasos; reconciliación de runs al arrancar; ejecución aislada para cancelación fuerte; pruebas de inferencia base versus adaptada; programación exclusiva de GPU; internacionalización completa; y una decisión explícita sobre plugins, artefactos y modelos soportados. Solo después tiene sentido ampliar el inventario hacia GGUF o una experiencia similar a LM Studio.
