import json
import base64
import io
import math

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
    page_title="Nube de Pensamiento",
    page_icon="💭",
    layout="wide"
)

st.title("💭 Nube de Pensamiento")

st.write(
    "Dibuja libremente tus ideas. La inteligencia artificial "
    "identificará los conceptos y creará una nube de pensamiento."
)


# ============================================================
# FUNCIONES
# ============================================================

def encode_image_to_base64(image):
    """
    Convierte una imagen PIL a Base64.
    """

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
    Limpia posibles bloques Markdown y extrae el JSON.
    """

    texto = texto.strip()

    if "```json" in texto:
        texto = texto.replace("```json", "")

    if "```" in texto:
        texto = texto.replace("```", "")

    texto = texto.strip()

    inicio = texto.find("{")
    fin = texto.rfind("}")

    if inicio != -1 and fin != -1:
        texto = texto[inicio:fin + 1]

    return texto


# ============================================================
# ANALIZAR DIBUJO
# ============================================================

def analizar_dibujo(client, image):

    base64_image = encode_image_to_base64(image)

    prompt = """
Analiza el dibujo realizado por el usuario.

El usuario está creando una NUBE DE PENSAMIENTO.

Tu tarea es identificar las ideas, palabras, conceptos y
relaciones que aparecen en el dibujo.

No necesitas interpretar literalmente cada línea del dibujo.
Concéntrate en detectar conceptos importantes.

Devuelve EXCLUSIVAMENTE un JSON válido.

Utiliza exactamente esta estructura:

{
    "tema_principal": "Tema principal",
    "ideas": [
        {
            "id": 1,
            "texto": "Idea",
            "importancia": 8,
            "padre": null
        },
        {
            "id": 2,
            "texto": "Subidea",
            "importancia": 5,
            "padre": 1
        }
    ]
}

REGLAS:

1. Debe existir un tema principal.
2. El tema principal debe tener "padre": null.
3. Las demás ideas deben indicar el ID de la idea relacionada.
4. "importancia" debe ser un número entre 1 y 10.
5. Utiliza máximo 20 ideas.
6. Mantén los textos cortos.
7. No inventes conceptos que no tengan relación razonable
   con el dibujo.
8. Si existen flechas o líneas entre conceptos, intenta
   utilizarlas para determinar las relaciones.
9. No escribas explicaciones fuera del JSON.
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
        max_tokens=2500,
        temperature=0.2
    )

    contenido = response.choices[0].message.content

    if not contenido:
        raise ValueError(
            "La inteligencia artificial no devolvió información."
        )

    contenido = limpiar_json(contenido)

    return json.loads(contenido)


# ============================================================
# CREAR NUBE DE PENSAMIENTO
# ============================================================

def crear_nube(datos, color_principal, color_secundario):

    ideas = datos.get("ideas", [])

    if not ideas:
        return None

    # --------------------------------------------------------
    # Buscar tema principal
    # --------------------------------------------------------

    principal = None

    for idea in ideas:
        if idea.get("padre") is None:
            principal = idea
            break

    if principal is None:
        principal = ideas[0]

    # --------------------------------------------------------
    # Crear diccionario
    # --------------------------------------------------------

    ideas_dict = {
        idea["id"]: idea
        for idea in ideas
    }

    hijos = {}

    for idea in ideas:

        padre = idea.get("padre")

        if padre not in hijos:
            hijos[padre] = []

        hijos[padre].append(idea)

    # --------------------------------------------------------
    # Posiciones
    # --------------------------------------------------------

    posiciones = {}

    posiciones[principal["id"]] = (0, 0)

    ideas_principales = hijos.get(
        principal["id"],
        []
    )

    cantidad = len(ideas_principales)

    if cantidad == 0:
        cantidad = 1

    # --------------------------------------------------------
    # Colocar ideas principales
    # --------------------------------------------------------

    radio = 5.0

    for i, idea in enumerate(ideas_principales):

        angulo = (
            2 * math.pi * i / cantidad
        )

        x = radio * math.cos(angulo)
        y = radio * math.sin(angulo)

        posiciones[idea["id"]] = (
            x,
            y
        )

        # --------------------------------------------
        # Subideas
        # --------------------------------------------

        subideas = hijos.get(
            idea["id"],
            []
        )

        cantidad_sub = len(subideas)

        if cantidad_sub == 0:
            continue

        for j, subidea in enumerate(subideas):

            separacion = 0.35

            angulo_sub = (
                angulo
                +
                (
                    j
                    -
                    (cantidad_sub - 1) / 2
                )
                * separacion
            )

            radio_sub = 2.5

            x2 = (
                x
                +
                radio_sub
                * math.cos(angulo_sub)
            )

            y2 = (
                y
                +
                radio_sub
                * math.sin(angulo_sub)
            )

            posiciones[subidea["id"]] = (
                x2,
                y2
            )

    # --------------------------------------------------------
    # Ideas que todavía no tengan posición
    # --------------------------------------------------------

    ids_sin_posicion = [
        idea["id"]
        for idea in ideas
        if idea["id"] not in posiciones
    ]

    for i, idea_id in enumerate(ids_sin_posicion):

        angulo = (
            2 * math.pi * i / max(
                len(ids_sin_posicion),
                1
            )
        )

        posiciones[idea_id] = (
            7 * math.cos(angulo),
            7 * math.sin(angulo)
        )

    # --------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 9)
    )

    ax.set_facecolor("#F8FAFC")

    # --------------------------------------------------------
    # Dibujar conexiones
    # --------------------------------------------------------

    for idea in ideas:

        padre = idea.get("padre")

        if padre is None:
            continue

        if idea["id"] not in posiciones:
            continue

        if padre not in posiciones:
            continue

        x1, y1 = posiciones[padre]
        x2, y2 = posiciones[idea["id"]]

        # Línea
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#CBD5E1",
            linewidth=2,
            alpha=0.8,
            zorder=1
        )

    # --------------------------------------------------------
    # Dibujar burbujas
    # --------------------------------------------------------

    for idea in ideas:

        idea_id = idea["id"]

        if idea_id not in posiciones:
            continue

        x, y = posiciones[idea_id]

        importancia = idea.get(
            "importancia",
            5
        )

        # ----------------------------------------------------
        # Tamaño
        # ----------------------------------------------------

        if idea_id == principal["id"]:

            size = 5000
            color = color_principal

        elif importancia >= 8:

            size = 3000
            color = "#8B5CF6"

        elif importancia >= 5:

            size = 2200
            color = color_secundario

        else:

            size = 1500
            color = "#94A3B8"

        # ----------------------------------------------------
        # Burbuja
        # ----------------------------------------------------

        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            alpha=0.90,
            edgecolors="white",
            linewidths=3,
            zorder=2
        )

        # ----------------------------------------------------
        # Texto
        # ----------------------------------------------------

        ax.text(
            x,
            y,
            idea["texto"],
            ha="center",
            va="center",
            fontsize=10 if idea_id != principal["id"] else 13,
            fontweight="bold",
            color="white",
            wrap=True,
            zorder=3
        )

    # --------------------------------------------------------
    # Título
    # --------------------------------------------------------

    titulo = datos.get(
        "tema_principal",
        "Nube de pensamiento"
    )

    ax.set_title(
        f"💭 {titulo}",
        fontsize=22,
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

    # --------------------------------------------------------
    # Tamaño
    # --------------------------------------------------------

    st.subheader("📐 Tamaño del tablero")

    canvas_width = st.slider(
        "Ancho",
        300,
        1200,
        700,
        50
    )

    canvas_height = st.slider(
        "Alto",
        250,
        800,
        450,
        50
    )

    # --------------------------------------------------------
    # Colores
    # --------------------------------------------------------

    st.subheader("🎨 Colores")

    background_color = st.color_picker(
        "Fondo",
        "#FFFFFF"
    )

    stroke_color = st.color_picker(
        "Color del lápiz",
        "#000000"
    )

    st.subheader("🖌️ Lápiz")

    stroke_width = st.slider(
        "Grosor",
        1,
        30,
        5
    )

    # --------------------------------------------------------
    # Colores de la nube
    # --------------------------------------------------------

    st.subheader("💭 Nube")

    cloud_main_color = st.color_picker(
        "Color principal",
        "#2563EB"
    )

    cloud_secondary_color = st.color_picker(
        "Color secundario",
        "#10B981"
    )

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔑 Inteligencia artificial")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    st.divider()

    st.caption(
        "Dibuja palabras, círculos, flechas y conexiones. "
        "La IA intentará convertir tu dibujo en una nube "
        "de pensamiento organizada."
    )


# ============================================================
# TABLERO
# ============================================================

st.subheader("✏️ Dibuja tus pensamientos")

canvas_result = st_canvas(

    fill_color="rgba(0,0,0,0)",

    stroke_width=stroke_width,

    stroke_color=stroke_color,

    background_color=background_color,

    height=canvas_height,

    width=canvas_width,

    drawing_mode="freedraw",

    key="thinking_canvas"
)


# ============================================================
# BOTONES
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:

    analizar = st.button(
        "💭 Crear nube de pensamiento",
        type="primary",
        use_container_width=True
    )

with col2:

    limpiar = st.button(
        "🗑️ Limpiar resultado",
        use_container_width=True
    )


if limpiar:

    if "nube" in st.session_state:
        del st.session_state["nube"]

    st.rerun()


# ============================================================
# ANALIZAR
# ============================================================

if analizar:

    if not api_key:

        st.error(
            "Introduce tu API Key en el panel lateral."
        )

        st.stop()

    if canvas_result.image_data is None:

        st.warning(
            "Primero dibuja algo en el tablero."
        )

        st.stop()

    # --------------------------------------------------------
    # Cliente
    # --------------------------------------------------------

    client = OpenAI(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Imagen
    # --------------------------------------------------------

    image_array = np.array(
        canvas_result.image_data
    )

    image = Image.fromarray(
        image_array.astype("uint8"),
        "RGBA"
    )

    # --------------------------------------------------------
    # IA
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Analizando tus pensamientos..."
    ):

        try:

            resultado = analizar_dibujo(
                client,
                image
            )

            st.session_state["nube"] = resultado

        except json.JSONDecodeError:

            st.error(
                "No fue posible interpretar la respuesta "
                "de la IA. Intenta nuevamente."
            )

        except Exception as e:

            st.error(
                f"Ocurrió un error: {e}"
            )


# ============================================================
# RESULTADO
# ============================================================

if "nube" in st.session_state:

    datos = st.session_state["nube"]

    st.divider()

    st.subheader("💭 Tu nube de pensamiento")

    st.write(
        f"### {datos.get('tema_principal', 'Sin título')}"
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    figura = crear_nube(
        datos,
        cloud_main_color,
        cloud_secondary_color
    )

    if figura:

        st.pyplot(
            figura,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Ideas detectadas
    # --------------------------------------------------------

    st.subheader("🧠 Ideas detectadas")

    ideas = datos.get(
        "ideas",
        []
    )

    for idea in ideas:

        importancia = idea.get(
            "importancia",
            5
        )

        st.write(
            f"**{idea['texto']}** "
            f"— importancia: {importancia}/10"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    with st.expander(
        "🔍 Ver información generada por la IA"
    ):

        st.json(datos)

    # --------------------------------------------------------
    # Descargar
    # --------------------------------------------------------

    json_data = json.dumps(
        datos,
        ensure_ascii=False,
        indent=4
    )

    st.download_button(
        "⬇️ Descargar nube",
        data=json_data,
        file_name="nube_pensamiento.json",
        mime="application/json"
    )

