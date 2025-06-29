
# 🧬 LaboAI

**LaboAI** est une plateforme intelligente de laboratoire numérique conçue pour la détection assistée par intelligence artificielle de plusieurs types de cancers. Cette solution intégrée combine plusieurs modèles spécialisés, notamment pour :

* 🩺 La classification d’images dermatologiques (cancer de la peau)
* 📊 L’analyse de données tabulaires issues d’examens médicaux (cancer du sein et cancer du poumon)

Le système repose sur une architecture modulaire robuste, comprenant :

* 🖥️ Un backend développé avec **Flask** pour l’inférence rapide et sécurisée des modèles d’IA
* 🌐 Une interface web ergonomique réalisée en **Next.js**, offrant aux professionnels de santé et chercheurs une expérience fluide pour interagir avec les outils de prédiction

---

## 🚀 Installation et démarrage rapide

1. 📥 Clonez ce dépôt sur votre machine.

2. 🖥️ Ouvrez deux terminaux distincts.

3. Dans le premier terminal, lancez le frontend :

```bash
cd frontend
npm install
npm run dev
```

4. Dans le second terminal, lancez le backend :

```bash
cd backend
pip install -r requirements.txt
python app.py
```

---

Cette organisation vous permet de développer, tester et utiliser facilement LaboAI, tout en tirant parti de la puissance des modèles d’IA pour la détection précoce des cancers. 🎯


