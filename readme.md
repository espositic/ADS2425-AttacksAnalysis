# Caso di Studio - Analisi dei Dati per la Sicurezza (ADS)

**Anno Accademico:** 2024/2025  
**Studente:** Matteo Esposito  
**Matricola:** 806075  
**Università:** Università degli Studi di Bari "Aldo Moro"

Questo progetto è un caso di studio per il corso di *Analisi dei Dati per la Sicurezza (ADS)*. 
---

## 📁 File principali

- `functions.py`: contiene tutte le funzioni utilizzate per caricare, processare, analizzare e modellare i dati.
- `main.py`: script principale che esegue l’intero flusso di preprocessing, training e valutazione.
- `trainDdosLabelNumeric.csv`: dataset per il training.
- `testDdosLabelNumeric.csv`: dataset per il test.

---

## ⚙️ Setup del progetto

### 1. Clona il repository (opzionale)
```bash
git clone https://github.com/espositic/ADS2425-AttacksAnalysis.git
cd ADS2425-AttacksAnalysis
```
### 2. Crea un ambiente virtuale
```bash
python -m venv venv
```
### 3. Attiva l’ambiente virtuale
#### Linux/macOS:
```bash
source venv/bin/activate
```
#### Windows:
```bash
venv\Scripts\activate
```
### 4. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 5. Esecuzione
```bash
python main.py
```