<div align="center">

<!-- BANNER -->
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Sentence--BERT-Multilingual-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="BERT"/>
<img src="https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite"/>
<img src="https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="sklearn"/>

<br/><br/>

# 🌾 LLM-NLP-AgroGraph-IA

### Motor de Búsqueda Semántica e Híbrida para Diagnóstico Fitosanitario

<p>
Sistema de <b>Minería de Texto</b> y <b>NLP</b> que combina <b>TF-IDF</b> (búsqueda léxica) con <b>Sentence-BERT</b> (búsqueda semántica) para recuperar documentos fitosanitarios relevantes a partir de descripciones de síntomas en cultivos agrícolas.
</p>

<br/>

| Componente | Tecnología | Propósito |
|:---:|:---:|:---|
| 📄 Almacén de Documentos | SQLite + TF-IDF | Ingesta, indexación y búsqueda léxica |
| 🧠 Búsqueda Semántica | Sentence-BERT | Comprensión del significado de la consulta |
| 🔀 Búsqueda Híbrida | TF-IDF + BERT | Fusión ponderada de ambos enfoques |
| 🌱 Gestión de Cultivos | SQLite | Configuración local de la parcela del usuario |
| 🗃️ Caché Top-K | SQLite | Resultados pre-calculados por enfermedad |

</div>

---

## 📑 Tabla de Contenidos

- [Visión General](#-visión-general)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Pipeline de Datos](#-pipeline-de-datos)
- [Módulos del Sistema](#-módulos-del-sistema)
  - [almacen_documentos.py](#-almacen_documentospy)
  - [busqueda_semantica.py](#-busqueda_semanticapy)
  - [mis_cultivos.py](#-mis_cultivospy)
- [Flujo de Búsqueda Híbrida](#-flujo-de-búsqueda-híbrida)
- [Diagrama de Secuencia](#-diagrama-de-secuencia)
- [Modelo de Datos](#-modelo-de-datos)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Uso](#-instalación-y-uso)
- [Tests](#-tests)
- [Stack Tecnológico](#-stack-tecnológico)

---

## 🔭 Visión General

**LLM-NLP-AgroGraph-IA** es el módulo de **Recuperación de Información (IR)** del ecosistema AgroGraph-MAS. Su objetivo principal es responder a la pregunta:

> *"Mi cultivo tiene [síntomas]. ¿Qué enfermedad podría ser y cómo la trato?"*

El sistema **no clasifica imágenes** (eso lo hace el módulo CNN del proyecto principal); en cambio, recibe un **diagnóstico textual** (nombre de enfermedad o descripción de síntomas) y recupera los documentos fitosanitarios más relevantes de una base de conocimiento local.

### ¿Por qué dos métodos de búsqueda?

| Método | Fortaleza | Debilidad |
|:---|:---|:---|
| **TF-IDF** (léxico) | Encuentra coincidencias exactas de términos técnicos | No entiende sinónimos ni paráfrasis |
| **Sentence-BERT** (semántico) | Comprende el *significado* de la consulta | Puede perder términos técnicos específicos |
| **Híbrido** (fusión) | Combina ambas fortalezas | — |

**Ejemplo:** La consulta *"polvo blanco harinoso en las hojas"* no menciona la palabra "oídio", pero el modelo semántico comprende que describe esa enfermedad y la posiciona en los resultados.

---

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    subgraph ENTRADA["📥 Entrada"]
        TXT["📄 Documentos .txt"]
        PDF["📄 Documentos .pdf"]
        USR["👤 Consulta del Usuario"]
    end

    subgraph INGESTA["⚙️ Pipeline de Ingesta"]
        PARSE["Parser de Metadatos<br/>(CULTIVO / ENFERMEDAD / FUENTE)"]
        SQLITE[("🗄️ SQLite<br/>tabla: documentos")]
        TFIDF_BUILD["Construir Índice<br/>TF-IDF Vectorizer"]
        BERT_BUILD["Construir Embeddings<br/>Sentence-BERT"]
        TFIDF_PKL["💾 tfidf.pkl"]
        EMB_PKL["💾 embeddings_bert.pkl"]
    end

    subgraph BUSQUEDA["🔍 Pipeline de Búsqueda"]
        TFIDF_SEARCH["Búsqueda TF-IDF<br/>Similitud Coseno"]
        BERT_SEARCH["Búsqueda BERT<br/>Similitud Coseno"]
        FUSION["🔀 Fusión Híbrida<br/>0.4 × TF-IDF + 0.6 × BERT"]
        FILTRO["🌱 Filtro por Cultivos<br/>(mis_cultivos)"]
        CACHE["🗃️ Caché Top-K"]
    end

    subgraph SALIDA["📤 Salida"]
        RESULTS["📋 Documentos Rankeados<br/>(id, cultivo, enfermedad,<br/>fuente, texto, scores)"]
    end

    TXT --> PARSE
    PDF --> PARSE
    PARSE --> SQLITE
    SQLITE --> TFIDF_BUILD
    SQLITE --> BERT_BUILD
    TFIDF_BUILD --> TFIDF_PKL
    BERT_BUILD --> EMB_PKL

    USR --> TFIDF_SEARCH
    USR --> BERT_SEARCH
    TFIDF_PKL --> TFIDF_SEARCH
    EMB_PKL --> BERT_SEARCH
    TFIDF_SEARCH --> FUSION
    BERT_SEARCH --> FUSION
    FUSION --> FILTRO
    FILTRO --> RESULTS
    RESULTS --> CACHE

    style ENTRADA fill:#1a1a2e,stroke:#16213e,color:#e0e0e0
    style INGESTA fill:#0f3460,stroke:#16213e,color:#e0e0e0
    style BUSQUEDA fill:#533483,stroke:#16213e,color:#e0e0e0
    style SALIDA fill:#e94560,stroke:#16213e,color:#e0e0e0
```

---

## 📊 Pipeline de Datos

### Fase 1 — Ingesta y Pre-procesamiento

```mermaid
flowchart LR
    A["📁 /documentos/"] -->|glob **/*.txt, **/*.pdf| B["cargar_desde_directorio()"]
    B -->|Extrae metadatos<br/>líneas 1-6| C{"¿Tiene CULTIVO<br/>y ENFERMEDAD?"}
    C -->|Sí| D["INSERT OR IGNORE<br/>en SQLite"]
    C -->|No| E["Fallback: usa<br/>nombre del archivo"]
    E --> D
    D -->|"Fase indexación"| F["construir_indice()"]
    D -->|"Fase embeddings"| G["construir_embeddings()"]
    F --> H["💾 tfidf.pkl<br/>(vectorizador + matriz TF-IDF)"]
    G --> I["💾 embeddings_bert.pkl<br/>(IDs + matriz de embeddings)"]

    style A fill:#2d3436,stroke:#636e72,color:#dfe6e9
    style D fill:#0984e3,stroke:#74b9ff,color:#fff
    style H fill:#00b894,stroke:#55efc4,color:#fff
    style I fill:#6c5ce7,stroke:#a29bfe,color:#fff
```

### Formato de los Documentos Fuente

Cada archivo `.txt` sigue esta estructura estandarizada:

```
CULTIVO: tomate
ENFERMEDAD: tizón tardío
FUENTE: Manual Fitosanitario SAGARPA 2019

DESCRIPCIÓN
[Texto descriptivo de la enfermedad...]

SÍNTOMAS
- [Lista de síntomas observables...]

CONDICIONES FAVORABLES
- [Factores ambientales que propician la enfermedad...]

TRATAMIENTO
1. [Pasos de tratamiento con productos y dosis...]

PREVENCIÓN
- [Medidas preventivas...]

PRODUCTOS RECOMENDADOS
- [Lista de productos comerciales...]
```

---

## 📦 Módulos del Sistema

### 📄 `almacen_documentos.py`

> **Responsabilidad:** Gestión del almacén de documentos fitosanitarios con SQLite + TF-IDF + Caché Top-K.

```mermaid
classDiagram
    class AlmacenDocumentos {
        +cargar_desde_directorio(directorio, ruta_bd) int
        +construir_indice(ruta_bd, ruta_tfidf) None
        +buscar(consulta, cultivos, top_k) list~dict~
        +guardar_topk(enfermedad, documentos) None
        +recuperar_topk(enfermedad) list~dict~
        +listar_documentos() list~dict~
        -_conectar(ruta_bd) Connection
        -_crear_tablas(con) None
        -_leer_metadatos_txt(texto) dict
        -_leer_pdf(ruta) str
        -_cargar_indice(ruta_tfidf) dict
    }

    class TfidfVectorizer {
        min_df: 1
        max_df: 0.95
        sublinear_tf: True
        ngram_range: 1~2
    }

    class SQLiteDB {
        documentos: tabla
        cache_topk: tabla
    }

    AlmacenDocumentos --> TfidfVectorizer : usa
    AlmacenDocumentos --> SQLiteDB : persiste en
```

#### Funciones clave

| Función | Descripción | Entrada | Salida |
|:---|:---|:---|:---|
| `cargar_desde_directorio()` | Parsea `.txt`/`.pdf`, extrae metadatos e inserta en SQLite | directorio, ruta_bd | `int` (docs insertados) |
| `construir_indice()` | Vectoriza textos con TF-IDF y serializa con pickle | ruta_bd, ruta_tfidf | `tfidf.pkl` en disco |
| `buscar()` | Vectoriza consulta, calcula similitud coseno, rankea | consulta, cultivos, top_k | `list[dict]` con scores |
| `guardar_topk()` | Almacena IDs de los mejores docs por enfermedad | enfermedad, docs | Escribe en `cache_topk` |
| `recuperar_topk()` | Recupera docs cacheados sin recalcular | enfermedad | `list[dict]` |

#### Parámetros del TF-IDF Vectorizer

```python
TfidfVectorizer(
    min_df=1,        # Incluir términos que aparezcan en al menos 1 documento
    max_df=0.95,     # Excluir términos en más del 95% de documentos (stop words)
    sublinear_tf=True,  # Aplicar 1 + log(tf) para suavizar frecuencias altas
    ngram_range=(1, 2),  # Unigramas y bigramas ("mancha", "mancha foliar")
)
```

---

### 🧠 `busqueda_semantica.py`

> **Responsabilidad:** Búsqueda semántica con Sentence-BERT y fusión híbrida (TF-IDF + BERT).

```mermaid
classDiagram
    class BusquedaSemantica {
        +construir_embeddings(ruta_bd, ruta_embeddings) None
        +buscar_semantico(consulta, cultivos, top_k) list~dict~
        +buscar_hibrido(consulta, cultivos, top_k, pesos) list~dict~
        -_obtener_modelo() SentenceTransformer
        -_cargar_embeddings(ruta) dict
    }

    class SentenceBERT {
        modelo: paraphrase-multilingual-MiniLM-L12-v2
        tipo: Multilingüe
        dimensiones: 384
        patron: Singleton
    }

    class FusionHibrida {
        peso_tfidf: 0.4
        peso_bert: 0.6
        formula: score = w1*tfidf + w2*bert
    }

    BusquedaSemantica --> SentenceBERT : singleton
    BusquedaSemantica --> FusionHibrida : calcula
    BusquedaSemantica ..> AlmacenDocumentos : importa índice TF-IDF
```

#### Modelo Sentence-BERT

| Propiedad | Valor |
|:---|:---|
| **Nombre** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Idiomas** | 50+ (incluye español) |
| **Dimensiones del embedding** | 384 |
| **Patrón de carga** | Singleton (se carga una sola vez en memoria) |
| **Uso** | Codifica documentos y consultas al mismo espacio vectorial |

#### Fórmula de Fusión Híbrida

```
score_híbrido = 0.4 × score_tfidf + 0.6 × score_bert
```

El peso mayor para BERT (0.6) refleja que la comprensión semántica es más valiosa que la coincidencia léxica para consultas en lenguaje natural agrícola.

---

### 🌱 `mis_cultivos.py`

> **Responsabilidad:** Gestión de la parcela del usuario (lista de cultivos para filtrado personalizado).

```mermaid
classDiagram
    class MisCultivos {
        +agregar(cultivo) bool
        +quitar(cultivo) bool
        +listar() list~str~
        +existe(cultivo) bool
        +limpiar_todo() None
        -_normalizar(cultivo) str
        -_conectar(ruta_bd) Connection
    }

    class Normalizacion {
        strip(): elimina espacios
        lower(): a minúsculas
        UNIQUE constraint: previene duplicados
    }

    MisCultivos --> Normalizacion : aplica
```

Este módulo permite al usuario registrar qué cultivos tiene en su parcela. Esos nombres se usan como **filtro** en las búsquedas: si el usuario solo tiene maíz y calabaza, los resultados de tomate se excluyen automáticamente.

---

## 🔀 Flujo de Búsqueda Híbrida

```mermaid
flowchart TB
    START(["👤 Usuario ingresa consulta:<br/>'manchas oscuras en hojas de tomate'"])

    subgraph RAMA_TFIDF["Canal Léxico (TF-IDF)"]
        T1["Cargar tfidf.pkl"]
        T2["Vectorizar consulta<br/>con TfidfVectorizer"]
        T3["Calcular similitud<br/>coseno vs. matriz"]
        T4["scores_tfidf: dict"]
    end

    subgraph RAMA_BERT["Canal Semántico (BERT)"]
        B1["Cargar embeddings_bert.pkl"]
        B2["Codificar consulta<br/>con SentenceTransformer"]
        B3["Calcular similitud<br/>coseno vs. embeddings"]
        B4["scores_bert: dict"]
    end

    MERGE["Unir doc_ids de ambos canales"]
    CALC["score = 0.4 × tfidf + 0.6 × bert"]
    SORT["Ordenar por score descendente"]
    FILTER{"¿Filtro por<br/>cultivos activo?"}
    YES["Excluir docs de cultivos<br/>no registrados"]
    TOPK["Limitar a top_k resultados"]
    OUT(["📋 Lista de documentos rankeados"])

    START --> T1 & B1
    T1 --> T2 --> T3 --> T4
    B1 --> B2 --> B3 --> B4
    T4 --> MERGE
    B4 --> MERGE
    MERGE --> CALC --> SORT --> FILTER
    FILTER -->|Sí| YES --> TOPK
    FILTER -->|No| TOPK
    TOPK --> OUT

    style RAMA_TFIDF fill:#0984e3,stroke:#74b9ff,color:#fff
    style RAMA_BERT fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style MERGE fill:#e17055,stroke:#fab1a0,color:#fff
    style CALC fill:#e17055,stroke:#fab1a0,color:#fff
```

---

## 🔄 Diagrama de Secuencia

### Flujo completo: Ingesta → Indexación → Búsqueda

```mermaid
sequenceDiagram
    autonumber
    participant U as 👤 Usuario
    participant AD as almacen_documentos
    participant DB as 🗄️ SQLite
    participant TF as TF-IDF Vectorizer
    participant BS as busqueda_semantica
    participant BERT as 🧠 Sentence-BERT
    participant FS as 💾 Filesystem

    Note over U,FS: ═══ FASE 1: Ingesta de Documentos ═══

    U->>AD: cargar_desde_directorio()
    AD->>FS: glob("**/*.txt", "**/*.pdf")
    FS-->>AD: Lista de archivos
    loop Cada archivo
        AD->>AD: _leer_metadatos_txt(texto)
        AD->>DB: INSERT OR IGNORE INTO documentos
    end
    AD-->>U: int (documentos insertados)

    Note over U,FS: ═══ FASE 2: Indexación ═══

    U->>AD: construir_indice()
    AD->>DB: SELECT id, texto FROM documentos
    DB-->>AD: filas
    AD->>TF: fit_transform(textos)
    TF-->>AD: matriz TF-IDF
    AD->>FS: pickle.dump → tfidf.pkl

    U->>BS: construir_embeddings()
    BS->>DB: SELECT id, texto FROM documentos
    DB-->>BS: filas
    BS->>BERT: model.encode(textos)
    BERT-->>BS: matriz de embeddings (N×384)
    BS->>FS: pickle.dump → embeddings_bert.pkl

    Note over U,FS: ═══ FASE 3: Búsqueda Híbrida ═══

    U->>BS: buscar_hibrido("manchas oscuras tomate")
    BS->>FS: cargar tfidf.pkl
    BS->>TF: transform(consulta)
    TF-->>BS: vector consulta TF-IDF
    BS->>BS: cosine_similarity → scores_tfidf

    BS->>FS: cargar embeddings_bert.pkl
    BS->>BERT: encode(consulta)
    BERT-->>BS: vector consulta BERT
    BS->>BS: cosine_similarity → scores_bert

    BS->>BS: score = 0.4×tfidf + 0.6×bert
    BS->>BS: sort + top_k + filtro cultivos
    BS->>DB: SELECT docs WHERE id IN (top_ids)
    DB-->>BS: documentos completos
    BS-->>U: list[dict] rankeados
```

---

## 🗄️ Modelo de Datos

```mermaid
erDiagram
    DOCUMENTOS {
        int id PK "AUTOINCREMENT"
        text cultivo "NOT NULL"
        text enfermedad "NOT NULL"
        text fuente "DEFAULT ''"
        text texto "NOT NULL"
    }

    CACHE_TOPK {
        int id PK "AUTOINCREMENT"
        text enfermedad "NOT NULL"
        text doc_ids "JSON array de IDs"
        text actualizado "datetime('now')"
    }

    MIS_CULTIVOS {
        int id PK "AUTOINCREMENT"
        text cultivo "NOT NULL UNIQUE"
        text agregado "datetime('now')"
    }

    DOCUMENTOS ||--o{ CACHE_TOPK : "referenciado por"
    MIS_CULTIVOS }o--|| DOCUMENTOS : "filtra búsquedas de"
```

### Constraints importantes

- `DOCUMENTOS` tiene un `UNIQUE(cultivo, enfermedad, fuente)` para evitar duplicados.
- `MIS_CULTIVOS` tiene `UNIQUE(cultivo)` y normalización a minúsculas.
- `CACHE_TOPK.doc_ids` almacena un JSON array: `[1, 3, 5]`.

---

## 📂 Estructura del Proyecto

```
LLM-NLP-AgroGraph-IA/
│
├── 📁 documentos/                    # Corpus de conocimiento fitosanitario
│   ├── mancha_foliar_maiz.txt        #   Maíz — Mancha foliar de turcicum
│   ├── oidio_calabaza.txt            #   Calabaza — Oídio (cenicilla)
│   └── tizon_tardio_tomate.txt       #   Tomate — Tizón tardío
│
├── 📁 datos/                         # Artefactos generados (gitignored)
│   ├── almacen.db                    #   Base de datos SQLite
│   ├── tfidf.pkl                     #   Índice TF-IDF serializado
│   └── embeddings_bert.pkl           #   Embeddings Sentence-BERT
│
├── 📁 modulos/                       # Código fuente principal
│   ├── __init__.py
│   ├── almacen_documentos.py         #   Ingesta, TF-IDF, caché Top-K
│   ├── busqueda_semantica.py         #   BERT semántico + fusión híbrida
│   └── mis_cultivos.py               #   Gestión de parcela del usuario
│
├── 📁 tests/                         # Suite de pruebas
│   ├── __init__.py
│   ├── test_almacen.py               #   6 tests: carga, índice, búsqueda, caché
│   ├── test_busqueda_semantica.py    #   4 tests: semántico, híbrido, filtro, pesos
│   └── test_mis_cultivos.py          #   6 tests: CRUD, normalización, limpieza
│
├── requirements.txt                  # Dependencias Python
└── .gitignore
```

---

## 🚀 Instalación y Uso

### Requisitos previos

- **Python 3.10+**
- **pip** (gestor de paquetes)

### 1. Clonar el repositorio

```bash
git clone https://github.com/wilber023/LLM-NLP-AgroGraph-IA.git
cd LLM-NLP-AgroGraph-IA
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Pipeline completo

```python
from modulos.almacen_documentos import cargar_desde_directorio, construir_indice
from modulos.busqueda_semantica import construir_embeddings, buscar_hibrido
from modulos.mis_cultivos import agregar

# ── Paso 1: Ingestar documentos ──
n = cargar_desde_directorio()
print(f"Documentos cargados: {n}")

# ── Paso 2: Construir índices ──
construir_indice()           # TF-IDF
construir_embeddings()       # Sentence-BERT (descarga modelo ~120 MB la primera vez)

# ── Paso 3: Configurar parcela (opcional) ──
agregar("tomate")
agregar("maíz")

# ── Paso 4: Buscar ──
resultados = buscar_hibrido(
    "manchas oscuras en hojas de tomate",
    cultivos=["tomate"],
    top_k=3,
)

for r in resultados:
    print(f"[{r['score_hibrido']:.4f}] {r['cultivo']} — {r['enfermedad']}")
    # [0.7823] tomate — tizón tardío
```

---

## 🧪 Tests

El proyecto incluye **16 tests** organizados por módulo:

```bash
# Ejecutar tests individualmente
python -m tests.test_almacen              # 6 tests
python -m tests.test_busqueda_semantica   # 4 tests
python -m tests.test_mis_cultivos         # 6 tests
```

| Test | Verifica |
|:---|:---|
| Carga de documentos | Parser de `.txt`, inserción en SQLite, deduplicación |
| Índice TF-IDF | Creación correcta del vectorizador y serialización |
| Búsqueda léxica | Ranking por similitud coseno, filtro por cultivo |
| Caché Top-K | Guardar y recuperar resultados pre-calculados |
| Búsqueda semántica | BERT encuentra "oídio" sin mencionar la palabra |
| Búsqueda híbrida | Fusión ponderada devuelve el resultado correcto |
| Fórmula híbrida | `score = 0.4×tfidf + 0.6×bert` se cumple |
| Gestión de cultivos | CRUD, normalización, UNIQUE constraint |

---

## 🛠️ Stack Tecnológico

<div align="center">

```mermaid
mindmap
  root((AgroGraph<br/>NLP Engine))
    NLP
      Sentence-BERT
        paraphrase-multilingual-MiniLM-L12-v2
        384 dimensiones
        50+ idiomas
      TF-IDF
        scikit-learn TfidfVectorizer
        Bigramas
        Sublinear TF
      NLTK
        Tokenización
        Preprocesamiento
    Almacenamiento
      SQLite
        documentos
        cache_topk
        mis_cultivos
      Pickle
        tfidf.pkl
        embeddings_bert.pkl
    ML Core
      PyTorch >= 2.0
      NumPy
      Cosine Similarity
    Ingesta
      pypdf
        Extracción de PDF
      Parser .txt
        Metadatos estructurados
```

</div>

---

## 📐 Decisiones de Diseño

### ¿Por qué Sentence-BERT y no un LLM completo para la búsqueda?

| Criterio | Sentence-BERT (MiniLM) | LLM (GPT/Llama) |
|:---|:---|:---|
| **Latencia** | ~5 ms por consulta | ~500-2000 ms |
| **Memoria** | ~120 MB | 4-16 GB |
| **Offline** | ✅ Funciona sin internet | ❌ Requiere API o GPU |
| **Calidad semántica** | Buena para IR | Excelente pero excesiva |

### ¿Por qué fusión híbrida y no solo BERT?

- **TF-IDF** captura términos técnicos exactos (ej. "Phytophthora infestans")
- **BERT** captura semántica (ej. "polvo blanco" → oídio)
- La **fusión 40/60** da prioridad a la comprensión semántica pero no ignora coincidencias léxicas exactas

### ¿Por qué SQLite y no Postgres/MongoDB?

- El sistema está diseñado para funcionar **embebido** en la app móvil AgroGraph-MAS
- **Zero-config**: no requiere servidor de base de datos
- Suficiente para el volumen de documentos fitosanitarios (~cientos)

---

<div align="center">

### Desarrollado como parte del ecosistema AgroGraph-MAS

<img src="https://img.shields.io/badge/Estado-En%20Desarrollo-yellow?style=flat-square" alt="Estado"/>
<img src="https://img.shields.io/badge/Licencia-MIT-green?style=flat-square" alt="Licencia"/>
<img src="https://img.shields.io/badge/Tests-16%20pasando-brightgreen?style=flat-square" alt="Tests"/>

<br/><br/>

*Motor de NLP para diagnóstico fitosanitario inteligente* 🌾🧠

</div>
