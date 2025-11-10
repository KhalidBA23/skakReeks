## SkakReeks

Et grafisk skakspil med AI-modstander, lavet i Python med pygame.

## Funktioner

- **Spil mod AI**: Vælg at spille som hvid eller sort.
- **Farvevalg**: Spil som hvid eller sort mod AI'en.
- **Sværhedsgrad**: Juster AI-styrke op til søgedybde 5.
- **Tidsbegrænsning**: AI'en har maksimalt 15 sekunder per træk.
- **Visuel feedback**: Se mulige træk, skak-advarsler og sidste træk.


## Installation og Kørsel

### Metode 1: Brug start.bat
Dobbeltklik på `start.bat` og tryk på "Run" hvis Windows spørger - det installerer automatisk pygame-ce og starter spillet.

### Metode 2: Manuel installation

1. Installer Python 3.12 eller 3.13 (Python 3.14 understøttes ikke endnu)
   Download fra: https://www.python.org/downloads/

2. Installer pygame-ce:
   ```
   pip install pygame-ce
   ```

3. Kør spillet:
   ```
   python skakBoard.py
   ```


## Krav

- Python 3.12 eller 3.13 (Python 3.14 understøttes ikke)
- pygame-ce (installeres automatisk med pip eller start.bat)

## Filstruktur

- `skakBoard.py` – Hovedfilen med GUI og spillogik.
- `alphabeta.py` – AI-algoritmen (Alpha-Beta pruning).
- `skakPieces.py` – Klasser for skakbrikker.
- `assets/` – Billeder til brikker og bræt.


