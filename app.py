import os
import json
import base64

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Tablero Inteligente",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Tablero Inteligente")
st.write(
    "Dibuja una idea, concepto o esquema y conviértelo automáticamente "
    "en un mapa mental utilizando inteligencia artificial."
)


# ============================================================
# FUNCIONES
# ============================================================

def encode_image_to_base64(image):
    """
    Convierte una imagen PIL a Base64.
    """
    import io

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def limpiar_json(texto):
    """
    Intenta extraer un JSON válido de la respuesta del modelo.
    """

    texto = texto.strip()

    # Eliminar posibles bloques Markdown
    if texto.startswith("```"):
        texto = texto.replace("```json", "")
        texto = texto.replace("```", "")
        texto = texto.strip()

    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio != -1 and fin != -1:
        texto = texto[inicio:fin + 1]

    return texto


def analizar_dibujo(client, image):
    """
    Envía el dibujo al modelo y solicita una estructura
    de mapa mental en JSON.
    """

    base64_image = encode_image_to_base64(image)

    prompt = """
Analiza el dibujo realizado por el usuario.

El objetivo es convertir el dibujo en un MAPA MENTAL.

Identifica:

1. El concepto principal.
2. Las ideas principales.
3. Las subideas.
4. Las relaciones entre las ideas.

No inventes demasiada información.
Utiliza principalmente lo que pueda observarse o inferirse
razonablemente del dibujo.

Devuelve EXCLUSIVAMENTE un JSON válido con esta estructura:

{
    "titulo": "Concepto principal",
    "nodos": [
        {
            "id": "1",
            "texto": "Idea principal",
            "padre": null
        },
        {
            "id": "2",
            "texto": "Subidea",
            "padre": "1"
        }
    ]
}

Reglas:

- Debe existir un nodo principal.
- El nodo principal debe tener padre null.
- Los demás nodos deben indicar el id de su padre.
- Usa textos cortos.
- Máximo 20 nodos.
- No incluyas explicaciones fuera del JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=2000,
        temperature=0.2
    )

    contenido = response.choices[0].message.content

    if not contenido:
        raise ValueError("La IA no devolvió información.")

    contenido = limpiar_json(contenido)

    return json.loads(contenido)


def construir_mapa_mental(datos):
    """
    Construye un mapa mental utilizando matplotlib.
    """

    nodos = datos.get("nodos", [])

    if not nodos:
        st.error("No se encontraron nodos.")
        return

    # --------------------------------------------------------
    # Buscar nodo principal
    # --------------------------------------------------------

    nodo_raiz = None

    for nodo in nodos:
        if nodo.get("padre") is None:
            nodo_raiz = nodo
            break

    if nodo_raiz is None:
        nodo_raiz = nodos[0]

    # --------------------------------------------------------
    # Construir relaciones padre -> hijos
    # --------------------------------------------------------

    hijos = {}

    for nodo in nodos:
        padre = nodo.get("padre")

        if padre not in hijos:
            hijos[padre] = []

        hijos[padre].append(nodo)

    # --------------------------------------------------------
    # Calcular posiciones
    # --------------------------------------------------------

    posiciones = {}

    posiciones[nodo_raiz["id"]] = (0, 0)

    # Nodos principales
    principales = hijos.get(nodo_raiz["id"], [])

    cantidad_principales = len(principales)

    if cantidad_principales == 0:
        cantidad_principales = 1

    radio_principal = 3.5

    for i, nodo in enumerate(principales):

        angulo = (
            2 * np.pi * i / cantidad_principales
        )

        x = radio_principal * np.cos(angulo)
        y = radio_principal * np.sin(angulo)

        posiciones[nodo["id"]] = (x, y)

        # ----------------------------------------------------
        # Subnodos
        # ----------------------------------------------------

        subnodos = hijos.get(nodo["id"], [])

        cantidad_sub = len(subnodos)

        if cantidad_sub == 0:
            continue

        radio_sub = 1.8

        for j, sub in enumerate(subnodos):

            angulo_sub = (
                angulo
                + (j - (cantidad_sub - 1) / 2)
                * 0.35
            )

            x2 = x + radio_sub * np.cos(angulo_sub)
            y2 = y + radio_sub * np.sin(angulo_sub)

            posiciones[sub["id"]] = (x2, y2)

    # --------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 9)
    )

    ax.set_facecolor("#F8FAFC")

    # --------------------------------------------------------
    # Dibujar conexiones
    # --------------------------------------------------------

    for nodo in nodos:

        padre = nodo.get("padre")

        if padre is None:
            continue

        if nodo["id"] not in posiciones:
            continue

        if padre not in posiciones:
            continue

        x1, y1 = posiciones[padre]
        x2, y2 = posiciones[nodo["id"]]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#94A3B8",
            linewidth=2,
            zorder=1
        )

    # --------------------------------------------------------
    # Dibujar nodos
    # --------------------------------------------------------

    for nodo in nodos:

        if nodo["id"] not in posiciones:
            continue

        x, y = posiciones[nodo["id"]]

        es_raiz = nodo["id"] == nodo_raiz["id"]

        if es_raiz:

            color = "#2563EB"
            tamaño = 2500

        else:

            color = "#10B981"
            tamaño = 1500

        ax.scatter(
            x,
            y,
            s=tamaño,
            color=color,
            edgecolors="white",
            linewidths=3,
            zorder=2
        )

        ax.text(
            x,
            y,
            nodo["texto"],
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            wrap=True,
            zorder=3
        )

    # --------------------------------------------------------
    # Configuración visual
    # --------------------------------------------------------

    titulo = datos.get(
        "titulo",
        "Mapa mental"
    )

    ax.set_title(
        f"🧠 {titulo}",
        fontsize=20,
        fontweight="bold",
        pad=20
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuración")

    st.subheader("📐 Tamaño")

    canvas_width = st.slider(
        "Ancho",
        min_value=300,
        max_value=1200,
        value=700,
        step=50
    )

    canvas_height = st.slider(
        "Alto",
        min_value=200,
        max_value=800,
        value=450,
        step=50
    )

    st.subheader("🎨 Apariencia")

    background_color = st.color_picker(
        "Color del fondo",
        "#FFFFFF"
    )

    stroke_color = st.color_picker(
        "Color del lápiz",
        "#000000"
    )

    stroke_width = st.slider(
        "Grosor del lápiz",
        min_value=1,
        max_value=30,
        value=5
    )

    st.divider()

    st.subheader("🔑 OpenAI")

    api_key = st.text_input(
        "API Key",
        type="password",
        help="Introduce tu clave de OpenAI."
    )

    st.divider()

    st.info(
        "Dibuja un concepto, esquema o mapa de ideas. "
        "Después presiona 'Crear mapa mental'."
    )


# ============================================================
# CANVAS
# ============================================================

st.subheader("🎨 Dibuja tu idea")

canvas_result = st_canvas(

    fill_color="rgba(0, 0, 0, 0)",

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=background_color,

    height=canvas_height,

    width=canvas_width,

    drawing_mode="freedraw",

    key="canvas"
)


# ============================================================
# BOTÓN
# ============================================================

col1, col2 = st.columns([1, 4])

with col1:

    analizar = st.button(
        "🧠 Crear mapa mental",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PROCESAMIENTO
# ============================================================

if analizar:

    # --------------------------------------------------------
    # Validar API Key
    # --------------------------------------------------------

    if not api_key:

        st.error(
            "Introduce tu API Key en el menú lateral."
        )

        st.stop()

    # --------------------------------------------------------
    # Validar dibujo
    # --------------------------------------------------------

    if canvas_result.image_data is None:

        st.warning(
            "Primero realiza un dibujo."
        )

        st.stop()

    # --------------------------------------------------------
    # Crear cliente
    # --------------------------------------------------------

    client = OpenAI(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Convertir canvas a imagen
    # --------------------------------------------------------

    input_numpy_array = np.array(
        canvas_result.image_data
    )

    input_image = Image.fromarray(
        input_numpy_array.astype("uint8"),
        "RGBA"
    )

    # --------------------------------------------------------
    # Mostrar proceso
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Analizando tu dibujo..."
    ):

        try:

            mapa = analizar_dibujo(
                client,
                input_image
            )

            st.session_state["mapa"] = mapa

        except json.JSONDecodeError:

            st.error(
                "La IA no devolvió un JSON válido. "
                "Intenta nuevamente."
            )

        except Exception as e:

            st.error(
                f"Ocurrió un error: {e}"
            )


# ============================================================
# MOSTRAR RESULTADO
# ============================================================

if "mapa" in st.session_state:

    mapa = st.session_state["mapa"]

    st.divider()

    st.subheader("🧠 Mapa mental generado")

    # --------------------------------------------------------
    # Información
    # --------------------------------------------------------

    st.write(
        f"**Concepto principal:** "
        f"{mapa.get('titulo', 'Sin título')}"
    )

    # --------------------------------------------------------
    # Mostrar gráfico
    # --------------------------------------------------------

    figura = construir_mapa_mental(
        mapa
    )

    if figura:

        st.pyplot(
            figura,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Mostrar estructura
    # --------------------------------------------------------

    with st.expander(
        "🔍 Ver estructura detectada"
    ):

        st.json(mapa)

    # --------------------------------------------------------
    # Descargar JSON
    # --------------------------------------------------------

    json_descarga = json.dumps(
        mapa,
        ensure_ascii=False,
        indent=4
    )

    st.download_button(
        label="⬇️ Descargar mapa mental",
        data=json_descarga,
        file_name="mapa_mental.json",
        mime="application/json"
    )
