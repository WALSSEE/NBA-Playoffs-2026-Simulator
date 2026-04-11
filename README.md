# 🏀 NBA Playoff Odds Calculator

Streamlit-sovellus NBA:n pudotuspelisarjojen todennäköisyyksien laskemiseen joukkueiden voimalukujen perusteella.

## Ominaisuudet

- **Sarja-analyysi** – Kahden joukkueen välisen sarjan todennäköisyydet
  - Sarjan voittotodennäköisyys
  - Marginaalilinjat: +3.5, +2.5, +1.5
  - Kaikki mahdolliset sarjatulokset (4-0, 4-1, 4-2, 4-3)
- **Konferenssi- & NBA-mestaruus** – Monte Carlo -simulaatio koko bracketille
  - Konferenssin mestaruustodennäköisyydet
  - NBA-mestaruustodennäköisyydet
- **Kotietu** – Säädettävä kotiedun vaikutus (oletus 3 pistettä)
- **Excel-tuonti** – Lataa omat voimaluvut Excel-tiedostosta

## Asennus paikallisesti

```bash
git clone https://github.com/SINUN_KÄYTTÄJÄNIMI/nba-playoff-odds.git
cd nba-playoff-odds
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Streamlit Cloudiin (GitHub)

1. **Luo GitHub-repositorio** ja push tämä koodi sinne
2. Mene osoitteeseen [share.streamlit.io](https://share.streamlit.io)
3. Kirjaudu sisään GitHub-tunnuksillasi
4. Klikkaa **"New app"**
5. Valitse repositorio, branch (`main`) ja tiedosto (`app.py`)
6. Klikkaa **"Deploy"** – valmis! 🎉

## Malli

Voitontodennäköisyys lasketaan logistisella funktiolla:

```
P(A voittaa) = 1 / (1 + exp(-diff × 0.155))
```

missä `diff = power_A + home_advantage - power_B`

- **Kotietu** lisätään kotijoukkueen voimalukuun jokaisessa kotipelissä
- NBA:n standardi kotipelijärjestys: 2-2-1-1-1 (pelit 1,2,5,7 paremmalla siemenellä)
- Konferenssi- ja NBA-mestaruudet lasketaan Monte Carlo -simulaatiolla (100 000 kierrosta)

## Voimalukulähteet

- **Net Rating** – NBA.com, ESPN
- **Adjusted Net Rating** – Cleaning the Glass
- **RAPTOR / EPM** – FiveThirtyEight, BBall-Index

## Excel-pohja

Lataa valmis Excel-pohja sovelluksen "Bracket"-välilehdeltä ja täytä joukkueiden tiedot:

| Joukkue | Konferenssi | Sija | Voima |
|---------|-------------|------|-------|
| Boston Celtics | Itä | 1 | 118.0 |
| ... | ... | ... | ... |

Lataa tiedosto sovelluksen sivupalkista.
