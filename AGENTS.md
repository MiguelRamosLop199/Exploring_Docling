# Explorando Docling

> Contexto para agentes LAID. Mantenido por el equipo; propuestas de IA marcadas como tales.

## Propósito
Sandbox de exploración y evaluación de [Docling](https://ds4sd.github.io/docling/) para la
documentación de proyectos de Data Science. Prueba conversión de documentos, chunking (jerárquico y
avanzado por límite de tokens) y OCR. Es un repositorio de análisis, no una librería publicada.

## Stack
- Lenguaje: Python (kernel de los notebooks 3.12.3; `.venv` contiene Python 3.14) (?)
- Frameworks: Docling (`docling`, `docling-core`, `docling-slim`), Transformers/HuggingFace, Pydantic (?)
- Almacén de datos: ninguno; PDF/DOCX de ejemplo en `resources/`
- Gestor de dependencias: pip + venv (`requirements.txt` creado en esta tarea)
- Versiones instaladas en `.venv` (Python 3.14.6), verificadas en `site-packages`: docling 2.131.0 ·
  docling-core 2.99.0 · docling-slim 2.131.0 · transformers 5.17.0 · tokenizers 0.23.2 · pydantic
  2.13.5 · torch 2.14.0 · pytest 9.1.1 (instalado, sin tests que lo usen)

## Comandos
- Instalar: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (?)
- Build: no aplica (no hay empaquetado)
- Test (todos): `.venv/bin/pytest -q` (tests de contrato en `tests/`)
- Test (uno): `.venv/bin/pytest -q tests/test_chunker.py`
- Lint / formato: no configurado (?)

## Estructura
- `docling_chunker.ipynb` — chunking avanzado (`HierarchicalChunker` + límite de tokens) sobre
  `resources/JerarquiaDocs.pdf`
- `docling_chunker_router.ipynb` — el mismo chunker aplicado a `resources/manuals/Archer AX11000.pdf`
- `basics.ipynb` — conversión, chunking básico y OCR
- `docling_explorer/chunker.py` — extracción de `MaxTokenLimitingChunker` y función
  `chunk_document(path, max_tokens) -> list[BaseChunk]`
- `tests/test_chunker.py` — tests de contrato del chunker
- `resources/` — documentos de prueba (`*.pdf`, `rel18/rel_14.docx`), salidas de ejemplo
  (`docling_outputs/`) e imágenes
- `README.md` — conclusiones del análisis

## Convenciones
- Código: notebooks exploratorios + módulos Python en `docling_explorer/`. Implementación de
  referencia extraída de la celda `MaxTokenLimitingChunker` de `docling_chunker.ipynb`
- Tests: `pytest`; fixtures y casos en `tests/`
- Commits: Conventional Commits + trailers LAID (skill `laid-commit-standard`)
- Ramas / PR: rama principal `main`; trabajo en rama con nombre estándar LAID (skill `laid-git-hygiene`)

## Contratos y arquitectura
- No hay contratos formales (OpenAPI, esquemas, protobuf) ni ADRs (?)
- Interfaz de referencia: `MaxTokenLimitingChunker.chunk(dl_doc) -> Iterator[BaseChunk]` (notebooks)
- Orden de construcción propuesto: firma/contrato de la función → pruebas de límite de tokens →
  implementación

## LAID
- Metodología: LAID v1.0 (skill `laid-methodology`) · AIWA: heredado de `laid-agents` (?)
- Tipo de proyecto: exploración / spike, sin producto desplegado (?)
- Modo por defecto: mixta · se decide por tarea con la rúbrica de verificabilidad
- Tracker: Jira (proyecto `LAID1`) configurado en `.laid/backlog-project.json`; la lectura de
  expedientes la realiza el subagente `tracker/jira-read`
- Delegable (`agente`/`mixta`): extracción de lógica de notebooks a módulos Python, `requirements.txt`,
  andamiaje, docs técnicas, tests de límite de tokens
- Solo manual: no hay rutas de auth/crypto/pagos; la interpretación de resultados de chunking y las
  conclusiones del análisis son manuales
- Datos: usar solo los documentos de `resources/`; no introducir datos reales de cliente · no se han
  identificado fixtures sintéticas (?)
- Orden de construcción: contrato de la función → pruebas → implementación
- Definición de hecho: criterios verificados con pruebas (`pytest`, pendiente (?) ) · revisión humana
  registrada (`Reviewed-by`) · cobertura mínima no acordada (?) · README/docs actualizadas · validación
  de quien pidió la tarea
- Validan: desarrollo (quien pide la extracción) (?) · sénior en modo `agente` (?) · calidad (?) · PO (?)
- Trazabilidad: commits con trailers LAID (skill `laid-commit-standard`); resúmenes marcados como
  propuesta de IA

## Preguntas abiertas
- Tipo de proyecto: ¿exploración puntual o base de una herramienta reutilizable?
- `requirements.txt`, tests y estructura de módulo creados en esta tarea; lint/CI siguen sin configurar
- `chunk_document(path, max_tokens)` devuelve `list[BaseChunk]` (decisión de esta tarea)
- ¿Versión de Python objetivo: 3.12 (kernel) o 3.14 (`.venv`)?
- ¿`resources/` contiene solo material público o hay documentos con licencia/confidencialidad?
- Cobertura mínima y validador humano sin acordar.
