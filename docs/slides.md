---
marp: true
theme: default
paginate: true
class: lead
---

<!--- view using:
npx @marp-team/marp-cli docs/slides.md --preview
--->

<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #FBF9F6;
  --surface: #FFFFFF;
  --text: #1A1614;
  --text-secondary: #6B6560;
  --primary: #1B2A3C;
  --accent: #B87333;
  --accent-subtle: #F0E6D8;
  --border: #E2DDD8;
  --font-heading: 'Playfair Display', Georgia, serif;
  --font-body: 'Inter', -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

section {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 2rem;
  line-height: 1.5;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 8% 12%;
}

h1, h2, h3, h4 {
  font-family: var(--font-heading);
  font-weight: 600;
  color: var(--text);
  line-height: 1.15;
  margin: 0;
}

h1 { font-size: 3.6rem; letter-spacing: -0.02em; }
h2 {
  font-size: 2.8rem;
  color: var(--primary);
}
h3 {
  font-family: var(--font-body);
  font-weight: 500;
  font-size: 1.8rem;
  color: var(--text-secondary);
  letter-spacing: 0.01em;
  margin-top: 0.4em;
}

/* Eyebrow accent rule under section headings */
h2::after {
  content: "";
  display: block;
  width: 3.5rem;
  height: 3px;
  margin: 0.6rem auto 0;
  background: var(--accent);
  border-radius: 2px;
}

p { color: var(--text-secondary); margin: 0.4em 0; }

a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid var(--accent-subtle);
}

strong { color: var(--text); font-weight: 600; }

table {
  font-size: 1.6rem;
  border-collapse: collapse;
  margin: 0 auto;
  background: var(--surface);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 24px rgba(26, 22, 20, 0.08);
}

th {
  background: var(--primary);
  color: var(--bg);
  font-family: var(--font-body);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 1.1rem;
  padding: 0.8rem 2rem;
}

td {
  padding: 0.7rem 2rem;
  border-top: 1px solid var(--border);
  color: var(--text);
}

tr td:first-child {
  font-family: var(--font-mono);
  color: var(--accent);
  font-weight: 500;
}

code {
  font-family: var(--font-mono);
  font-size: 1.6rem;
  background: var(--accent-subtle);
  color: var(--primary);
  padding: 0.15em 0.5em;
  border-radius: 6px;
}

pre {
  background: var(--primary);
  border-radius: 12px;
  padding: 1.2rem 1.8rem;
  box-shadow: 0 4px 24px rgba(26, 22, 20, 0.12);
}
pre code {
  background: transparent;
  color: var(--bg);
  font-size: 1.4rem;
}

section::after {
  color: var(--text-muted, #9C9590);
  font-family: var(--font-mono);
  font-size: 0.9rem;
}

/* Brand logo overlay, lower-right */
section::before {
  content: "";
  position: absolute;
  right: 2.5%;
  bottom: 2.5%;
  width: 130px;
  height: 56px;
  background-image: url('Chattanooga_AIC_Negative_Logo_Light.svg');
  background-repeat: no-repeat;
  background-position: bottom right;
  background-size: contain;
  opacity: 0.85;
  pointer-events: none;
}

/* Title slide */
section.lead h1 { font-size: 4.2rem; }
section.lead h3 { font-size: 2rem; }
</style>

# Train Language Models from Scratch

### Build intuition, write the code, ship a model

Groups of 3 · GPUs provided · ~1 min per run

---

## Agenda

| Time | Topic |
|------|-------|
| 1:00p | Refreshments & networking |
| 1:30p | Form groups |
| 1:35p | Welcome & Game of Life |
| 1:50p | Setup & GPU access |
| 2:00p | Tokenization |
| 2:25p | Transformer |

---

## Agenda - continued

| Time | Topic |
|------|-------|
| 3:00p | Ice cream |
| 3:10p | Training loop |
| 3:35p | Launch training |
| 3:45p | Text generation |
| 4:10p | Putting it together |
| 4:30p | Competition: best AI poet |
| 4:55p | Wrap-up & Q&A |

---

## Emergence

### Simple rules → surprising behavior

Game of Life ↔ a transformer at scale.

[youtu.be/C2vgICfQawE](https://youtu.be/C2vgICfQawE)

---

## How we work

### Short lecture, then build

GPUs remove the wait.

You end with a model that writes Shakespeare.

---

## Setup & GPU access

`README.md`

`env.sample`

---

## 1 · Tokenization

### text → numbers

`docs/01-tokenization.md`

`model.py`

[Visualization: Tokenization Live](visualizations/tokenization_live.html)

---

## 2 · The transformer

### embeddings · attention · MLP

`docs/02-the-transformer.md`

`model.py`

[Visualization: Transformer Flow](visualizations/transformer_flow.html)

---

## 3 · Training loop

### loss · backprop · AdamW

`docs/03-training-loop.md`

`train.py`

[Visualization: Step Cycle](visualizations/training_loop_stepcycle.html)
[Visualization: Loss Curves](visualizations/training_loop_losscurves.html)

---

## Launch training

```shell
uv run train.py --batch_size 512 --max_steps 625
```

~1 min on the queue.

---

## 4 · Text generation

### temperature · top-k · decoding

`docs/04-text-generation.md`

`generate.py`

---

## 5 · Putting it together

### loss curves · scaling experiments

`train.py`

---

## 6 · Competition

### Best AI poet

Scale up · find data · submit your best poem

---

## Wrap-up

### Everyone shipped a model

Submit results after the workshop.

Winners revealed next month.

---

# Thank you

### Questions?

---

# Resources

Club Website: https://ainooga.org
Presenter: John Goodwin

## Special Thanks

- The Enterprise Center: https://theenterprisecenter.com
- Ryann Jones
