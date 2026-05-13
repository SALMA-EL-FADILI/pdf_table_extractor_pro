from pcm_extractor import extract_pdf

result = extract_pdf(r"C:\Users\Hp\Downloads\IB_Maroc_2S05.pdf")

if result["success"]:
    print(f"✅ OK — {result['output_path']}")
    print(f"   Score qualité : {result['avg_quality_score']:.0f}/100")
    print(f"   Durée         : {result['duration']:.1f}s")
    for s in result["sections"]:
        print(f"   - {s['name']} (page {s['page']}) → {s['rows']} lignes")
else:
    print(f"❌ Erreur : {result['error']}")