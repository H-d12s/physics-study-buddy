import os
import math
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb


os.environ["GROQ_API_KEY"] = "apikey here"
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


embedder = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    {"id": "doc_001", "topic": "Newton's Laws of Motion", "text": """Newton's three laws of motion form the foundation of classical mechanics.

First Law (Law of Inertia): An object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted upon by a net external force. This means objects resist changes to their state of motion.

Second Law (F = ma): The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. Mathematically: F = ma, where F is force in Newtons (N), m is mass in kilograms (kg), and a is acceleration in m/s².

Third Law (Action-Reaction): For every action, there is an equal and opposite reaction. If object A exerts a force on object B, then object B exerts an equal and opposite force on object A.

Examples:
- A book resting on a table (First Law)
- Pushing a trolley — heavier trolley needs more force to accelerate (Second Law)
- Rocket propulsion — gas expelled downward pushes rocket upward (Third Law)

Key units: Force (N = kg·m/s²), Mass (kg), Acceleration (m/s²)."""},

    {'id': 'doc_002', 'topic': 'Kinematics and Equations of Motion', 'text': """Kinematics is the study of motion without considering its causes. It deals with displacement, velocity, acceleration, and time.

Key Definitions:
- Displacement (s): Change in position (vector quantity, in metres)
- Velocity (v): Rate of change of displacement (m/s)
- Acceleration (a): Rate of change of velocity (m/s²)

Four Equations of Motion (for uniform acceleration):
1. v = u + at
2. s = ut + ½at²
3. v² = u² + 2as
4. s = ((u + v) / 2) × t

Where: u = initial velocity, v = final velocity, a = acceleration, t = time, s = displacement.

Free Fall: When an object falls under gravity alone, a = g = 9.8 m/s² (downward). Initial velocity u = 0 if dropped from rest.

Projectile Motion: An object launched at an angle follows a curved path. Horizontal velocity is constant; vertical velocity changes due to gravity.

Example: A ball is dropped from rest and falls for 3 seconds.
s = ut + ½at² = 0 + ½ × 9.8 × 9 = 44.1 metres."""},

    {'id': 'doc_003', 'topic': 'Work, Energy, and Power', 'text': """Work, energy, and power are central concepts in mechanics.

Work (W): Work is done when a force causes displacement.
Formula: W = F × d × cos(θ)
Where F = force (N), d = displacement (m), θ = angle between force and displacement.
Unit: Joule (J). If force and displacement are in the same direction, θ = 0° and W = F × d.

Kinetic Energy (KE): Energy possessed by a moving object.
Formula: KE = ½mv²
Where m = mass (kg), v = velocity (m/s).

Potential Energy (PE): Energy stored due to position.
Gravitational PE: PE = mgh
Where m = mass (kg), g = 9.8 m/s², h = height (m).

Work-Energy Theorem: The net work done on an object equals its change in kinetic energy.
W_net = ΔKE = KE_final − KE_initial

Conservation of Energy: Total mechanical energy (KE + PE) is conserved in the absence of friction.

Power (P): Rate of doing work.
Formula: P = W / t = F × v
Unit: Watt (W = J/s).

Example: A 2 kg ball moving at 5 m/s has KE = ½ × 2 × 25 = 25 J."""},

    {'id': 'doc_004', 'topic': 'Gravitation', 'text': """Gravitation is the universal force of attraction between all objects with mass.

Newton's Law of Universal Gravitation:
F = G × (m₁ × m₂) / r²
Where G = 6.674 × 10⁻¹¹ N·m²/kg² (Universal Gravitational Constant), m₁ and m₂ are masses (kg), r = distance between centres (m).

Acceleration due to Gravity (g):
On Earth's surface, g ≈ 9.8 m/s². It varies slightly with altitude and latitude.
g = GM/R² where M = Earth's mass, R = Earth's radius.

Gravitational Potential Energy:
PE = −GMm/r (negative because gravity is attractive)
Near Earth's surface: PE = mgh

Escape Velocity: Minimum velocity needed to escape a planet's gravity.
v_escape = √(2GM/R)
For Earth: v_escape ≈ 11.2 km/s

Orbital Velocity: Velocity needed to maintain circular orbit.
v_orbital = √(GM/r)

Kepler's Laws:
1. Planets move in ellipses with the sun at one focus.
2. Equal areas are swept in equal times.
3. T² ∝ r³ (orbital period squared is proportional to orbital radius cubed).

Example: Calculate g on a planet with mass 2M and radius 2R of Earth.
g_planet = G(2M)/(2R)² = GM/(2R²) = g_earth/2 ≈ 4.9 m/s²."""},

    {'id': 'doc_005', 'topic': 'Thermodynamics', 'text': """Thermodynamics studies heat, temperature, and energy transfer.

Basic Concepts:
- Temperature: Measure of average kinetic energy of particles (in Kelvin: K = °C + 273)
- Heat (Q): Energy transferred due to temperature difference (Joules)
- Internal Energy (U): Total kinetic + potential energy of all particles in a system

Laws of Thermodynamics:

Zeroth Law: If A is in thermal equilibrium with B, and B with C, then A is in equilibrium with C. (Basis for temperature measurement)

First Law (Conservation of Energy):
ΔU = Q − W
Heat added to a system increases internal energy; work done by the system decreases it.

Second Law: Heat flows naturally from hot to cold. Entropy (disorder) of an isolated system always increases. No engine is 100% efficient.

Third Law: As temperature approaches absolute zero (0 K), entropy approaches a minimum constant value.

Heat Transfer Methods:
- Conduction: Through direct contact (solids)
- Convection: Through fluid movement (liquids, gases)
- Radiation: Through electromagnetic waves (no medium needed)

Specific Heat Capacity (c): Heat required to raise 1 kg by 1°C.
Q = mcΔT

Example: Heat needed to raise 2 kg of water by 10°C:
Q = 2 × 4200 × 10 = 84,000 J (c_water = 4200 J/kg°C)."""},

    {'id': 'doc_006', 'topic': 'Waves and Oscillations', 'text': """Waves transfer energy from one place to another without transferring matter.

Types of Waves:
- Transverse Waves: Oscillation perpendicular to direction of travel (light, water waves)
- Longitudinal Waves: Oscillation parallel to direction of travel (sound)

Key Wave Properties:
- Amplitude (A): Maximum displacement from equilibrium
- Wavelength (λ): Distance between two consecutive crests (metres)
- Frequency (f): Number of complete cycles per second (Hertz, Hz)
- Period (T): Time for one complete cycle. T = 1/f
- Wave Speed (v): v = f × λ

Simple Harmonic Motion (SHM): Oscillation where restoring force is proportional to displacement.
Examples: Simple pendulum, mass-spring system.

For a simple pendulum: T = 2π√(L/g)
Where L = length (m), g = 9.8 m/s²

For a spring-mass system: T = 2π√(m/k)
Where m = mass (kg), k = spring constant (N/m)

Resonance: When driving frequency equals natural frequency, amplitude increases dramatically.

Sound Waves:
- Speed of sound in air ≈ 343 m/s at 20°C
- Loudness measured in decibels (dB)
- Pitch related to frequency

Example: A wave has frequency 200 Hz and wavelength 1.5 m.
Speed = f × λ = 200 × 1.5 = 300 m/s."""},

    {'id': 'doc_007', 'topic': 'Electrostatics', 'text': """Electrostatics deals with electric charges at rest and the forces between them.

Electric Charge:
- Two types: positive (+) and negative (−)
- Like charges repel; opposite charges attract
- Unit: Coulomb (C)
- Charge is quantised: q = ne, where e = 1.6 × 10⁻¹⁹ C (charge of one electron)

Coulomb's Law:
F = k × (q₁ × q₂) / r²
Where k = 8.99 × 10⁹ N·m²/C² (Coulomb's constant), q₁ and q₂ are charges, r = distance.

Electric Field (E): Force per unit charge.
E = F/q = kQ/r²
Unit: N/C or V/m. Field lines point away from positive, toward negative charges.

Electric Potential (V): Work done per unit charge to bring a positive charge from infinity.
V = kQ/r
Unit: Volt (V)

Relationship: E = −dV/dr (field is negative gradient of potential)

Capacitance (C): Ability to store charge.
C = Q/V
Unit: Farad (F)
For a parallel plate capacitor: C = ε₀A/d

Electric Potential Energy:
U = kq₁q₂/r

Example: Two charges of 3μC and 4μC are 0.1m apart.
F = 9×10⁹ × 3×10⁻⁶ × 4×10⁻⁶ / 0.01 = 10.8 N."""},

    {'id': 'doc_008', 'topic': 'Current Electricity', 'text': """Current electricity deals with the flow of electric charges through conductors.

Electric Current (I): Rate of flow of charge.
I = Q/t
Unit: Ampere (A). Conventional current flows from + to −.

Ohm's Law: V = IR
Where V = voltage (Volts), I = current (Amperes), R = resistance (Ohms, Ω)

Resistance (R): Opposition to current flow.
R = ρL/A
Where ρ = resistivity (Ω·m), L = length, A = cross-sectional area.

Resistors in Series: R_total = R₁ + R₂ + R₃
Resistors in Parallel: 1/R_total = 1/R₁ + 1/R₂ + 1/R₃

Electric Power:
P = VI = I²R = V²/R
Unit: Watt (W)

Electric Energy:
E = P × t = VIt
Unit: Joule (J) or kilowatt-hour (kWh) for practical use.

Kirchhoff's Laws:
1. Junction Rule (KCL): Sum of currents entering a junction = sum leaving.
2. Loop Rule (KVL): Sum of voltages around any closed loop = 0.

Electromotive Force (EMF): Energy supplied per unit charge by a source.
Terminal voltage = EMF − I × internal resistance

Example: A 12V battery connected to a 4Ω resistor.
I = V/R = 12/4 = 3A
P = I²R = 9 × 4 = 36W"""},

    {'id': 'doc_009', 'topic': 'Optics — Ray and Wave', 'text': """Optics is the study of light and its behaviour.

Ray Optics (Geometrical Optics):

Reflection: Angle of incidence = Angle of reflection. Occurs at smooth surfaces.
- Plane mirror: Image is virtual, erect, laterally inverted, same size, same distance behind mirror.
- Concave mirror: Can form real or virtual images. Used in torches, telescopes.
- Convex mirror: Always forms virtual, erect, diminished images. Used as rear-view mirrors.

Mirror Formula: 1/f = 1/v + 1/u
Magnification: m = −v/u

Refraction: Bending of light as it passes from one medium to another.
Snell's Law: n₁ sin(θ₁) = n₂ sin(θ₂)
Refractive Index: n = speed of light in vacuum / speed in medium = c/v

Total Internal Reflection: When light hits the boundary at an angle greater than the critical angle, it reflects entirely back. Basis of optical fibres.

Lenses:
- Convex (converging) lens: Thicker at centre. Focuses light.
- Concave (diverging) lens: Thinner at centre. Spreads light.
Lens Formula: 1/f = 1/v − 1/u
Power of lens: P = 1/f (in Dioptres, D)

Wave Optics:
- Interference: Superposition of two coherent waves (Young's double slit experiment)
- Diffraction: Bending of waves around obstacles
- Polarisation: Restriction of wave oscillation to one plane (applies to transverse waves only)

Example: A convex lens has focal length 0.2m. Power = 1/0.2 = 5D."""},

    {'id': 'doc_010', 'topic': 'Modern Physics — Photoelectric Effect and Atomic Models', 'text': """Modern physics covers phenomena that cannot be explained by classical physics.

Photoelectric Effect:
When light of sufficient frequency hits a metal surface, electrons are emitted.
Key observations:
- Emission depends on frequency, not intensity.
- There is a minimum frequency called threshold frequency (f₀) below which no emission occurs.
- Kinetic energy of emitted electrons: KE = hf − φ
Where h = 6.626 × 10⁻³⁴ J·s (Planck's constant), f = frequency, φ = work function.
Explained by Einstein using photon model — light consists of quanta called photons.
Energy of one photon: E = hf

Atomic Models:
1. Thomson's Model (Plum Pudding): Electrons embedded in positive sphere. Disproved by Rutherford.
2. Rutherford's Model: Nucleus at centre; electrons orbit around it. Could not explain stability.
3. Bohr's Model: Electrons orbit in fixed energy levels. Energy emitted/absorbed when electrons jump levels.
   Energy of nth orbit: Eₙ = −13.6/n² eV (for hydrogen)

De Broglie Hypothesis: All matter has wave-like properties.
Wavelength: λ = h/mv (h = Planck's constant, m = mass, v = velocity)

Nuclear Physics Basics:
- Nucleus contains protons and neutrons
- Radioactive decay: Alpha (α), Beta (β), Gamma (γ) emission
- Half-life: Time for half the nuclei to decay
- Mass-energy equivalence: E = mc²

Example: Calculate energy of photon with frequency 5×10¹⁴ Hz.
E = hf = 6.626×10⁻³⁴ × 5×10¹⁴ = 3.31×10⁻¹⁹ J."""}
]


client = chromadb.Client()
collection = client.create_collection("physics_kb")
collection.add(
    documents=[d['text'] for d in documents],
    embeddings=embedder.encode([d['text'] for d in documents]).tolist(),
    ids=[d['id'] for d in documents],
    metadatas=[{'topic': d['topic']} for d in documents]
)


class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2



def memory_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    question = state["question"]
    user_name = state.get("user_name", "")

    if "my name is" in question.lower():
        phrase = question.lower().split("my name is")[-1]
        words = phrase.strip().split()
        if words:
            user_name = words[0].strip(".,!?").capitalize()

    messages = messages[-6:]
    messages.append({"role": "user", "content": question})
    return {"messages": messages, "user_name": user_name, "eval_retries": 0}


def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    prompt = f"""You are a router for a Physics Study Buddy assistant.
Based on the student's question, choose ONE route only:

- retrieve: Question is about a Physics concept, formula, law, or theory (use the knowledge base)
- tool: Question requires a calculation or arithmetic (use the calculator)
- memory_only: Simple greeting, thank you, or question about the conversation itself

Student question: {question}

Reply with ONE word only: retrieve, tool, or memory_only"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in ["retrieve", "tool", "memory_only"]:
        route = "retrieve"
    return {"route": route}


def retrieval_node(state: CapstoneState) -> dict:
    question = state["question"]
    results = collection.query(
        query_embeddings=embedder.encode([question]).tolist(),
        n_results=3
    )
    chunks = results["documents"][0]
    topics = [m["topic"] for m in results["metadatas"][0]]
    context = ""
    for topic, chunk in zip(topics, chunks):
        context += f"[{topic}]\n{chunk}\n\n"
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    question = state["question"]
    try:
        prompt = f"""Extract a single mathematical expression from this physics question and return ONLY the expression, nothing else.
Question: {question}
Expression:"""
        response = llm.invoke(prompt)
        expression = response.content.strip()
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        tool_result = f"Calculator result: {expression} = {round(result, 4)}"
    except Exception as e:
        tool_result = f"Calculator could not compute this. Please check your expression. (Error: {str(e)})"
    return {"tool_result": tool_result}


def answer_node(state: CapstoneState) -> dict:
    question = state["question"]
    retrieved = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages = state.get("messages", [])
    user_name = state.get("user_name", "")
    eval_retries = state.get("eval_retries", 0)

    name_str = f" The student's name is {user_name}." if user_name else ""
    retry_str = ""
    if eval_retries > 0:
        retry_str = "Your previous answer scored low on faithfulness. Answer STRICTLY using only the context below — do not add anything from outside."

    context_section = ""
    if retrieved:
        context_section = f"\nKNOWLEDGE BASE CONTEXT:\n{retrieved}"
    if tool_result:
        context_section += f"\nCALCULATOR RESULT:\n{tool_result}"

    history_str = ""
    for msg in messages[-4:]:
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""You are a helpful Physics Study Buddy for B.Tech students.{name_str}
{retry_str}

STRICT RULES:
- Answer ONLY from the KNOWLEDGE BASE CONTEXT or CALCULATOR RESULT provided below.
- If the answer is not in the context, say: "I don't have information on that topic yet. Please refer to your textbook or ask your professor."
- Never make up formulas, values, or facts.
- Be clear, friendly, and educational.
- If a student's name is known, use it naturally.

CONVERSATION HISTORY:
{history_str}

{context_section}

Student question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return {"answer": response.content.strip()}


def eval_node(state: CapstoneState) -> dict:
    answer = state.get("answer", "")
    retrieved = state.get("retrieved", "")
    eval_retries = state.get("eval_retries", 0)

    if not retrieved:
        return {"faithfulness": 1.0, "eval_retries": eval_retries}

    prompt = f"""Rate how faithful this answer is to the provided context.
Score 0.0 to 1.0 — where 1.0 means the answer uses ONLY information from the context.

CONTEXT:
{retrieved[:1000]}

ANSWER:
{answer}

Reply with a single decimal number only (e.g. 0.8):"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except:
        score = 0.8

    eval_retries += 1
    return {"faithfulness": score, "eval_retries": eval_retries}


def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    answer = state.get("answer", "")
    messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}



def route_decision(state: CapstoneState) -> str:
    return state.get("route", "retrieve")

def eval_decision(state: CapstoneState) -> str:
    faithfulness = state.get("faithfulness", 1.0)
    eval_retries = state.get("eval_retries", 0)
    if faithfulness < FAITHFULNESS_THRESHOLD and eval_retries < MAX_EVAL_RETRIES:
        return "answer"
    return "save"



graph = StateGraph(CapstoneState)

graph.add_node("memory", memory_node)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieval_node)
graph.add_node("skip", skip_retrieval_node)
graph.add_node("tool", tool_node)
graph.add_node("answer", answer_node)
graph.add_node("eval", eval_node)
graph.add_node("save", save_node)

graph.set_entry_point("memory")

graph.add_edge("memory", "router")
graph.add_edge("retrieve", "answer")
graph.add_edge("skip", "answer")
graph.add_edge("tool", "answer")
graph.add_edge("answer", "eval")
graph.add_edge("save", END)

graph.add_conditional_edges("router", route_decision, {
    "retrieve": "retrieve",
    "tool": "tool",
    "memory_only": "skip"
})

graph.add_conditional_edges("eval", eval_decision, {
    "answer": "answer",
    "save": "save"
})

app = graph.compile(checkpointer=MemorySaver())

__all__ = ["app", "CapstoneState"]