Spilmuligheder
Farvevalg: Spil som hvid eller sort mod AI'en

Sværhedsgrad: Juster AI-styrke op til søgedybde 5

Tidsbegrænsning: AI'en har maksimalt 15 sekunder per træk

Funktioner
Undo-knap: Fortryd dit sidste træk (og AI'ens)

Pause-funktion: Tryk P for at pause spillet

Debug-mode: I pause kan du flytte brikker frit på brættet

Visuel feedback: Se mulige træk, skak-advarsler og sidste træk

Installation & Kørsel (alt i terminal)
Klon projektet:
git clone https://github.com/KhalidBA23/skakReeks.git
cd skakReeks

Installer PyInstaller:
pip install pyinstaller

Byg .exe-filen:
pyinstaller --noconfirm --onefile --windowed skakBoard.py

Kør spillet:
cd dist
.\skakBoard.exe (brug ./skakBoard.exe hvis du er i Bash)

