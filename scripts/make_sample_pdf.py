"""Write ``src/optivision/easy/data/sample.pdf``, the 20-page sample document of easy mode.

Easy mode's wiring check (FINAL_DESIGN 5.5, G-wire) searches pages with a sentence taken
from each of them. When that fails on a user's own pages, it runs once more on this file:
if the model finds these pages, the model is wired correctly and the user's pages look alike.
The notebook and the docs build an index of it as a trial run.

Every word of it is written for this package (fictional places and people; no text, image or
data is copied from anywhere) and it is released under the package's MIT licence. Each page
is a short document on its own subject, so each holds sentences no other page has. Only the
standard PDF fonts are used (nothing embedded), which keeps the file small.

    python scripts/make_sample_pdf.py            # rewrite the file (needs reportlab)
    python scripts/make_sample_pdf.py --check    # exit 1 if the file's text differs from this script
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "optivision" / "easy" / "data" / "sample.pdf"

#: (title, paragraphs, table header, table rows) per page; written for this package, MIT
PAGES: list[tuple[str, list[str], list[str], list[list[str]]]] = [
    ("Harbour Water Quality, Spring Survey", [
        "The harbour survey team sampled eleven buoys along the northern breakwater during the April low tides.",
        ("Dissolved oxygen stayed above the seasonal threshold at every buoy except the one beside the old fish "
        "market. Turbidity rose sharply after the storm on the ninth, when silt from the Kestrel River reached "
        "the inner basin."),
        "The council will repeat the sampling in June and post the readings on the noticeboard by the ferry steps.",
    ], ["Buoy", "Oxygen (mg/L)", "Turbidity (NTU)"], [["N1", "8.4", "3.1"], ["N5", "7.9", "4.6"],
                                                      ["Fish market", "5.2", "9.8"]]),
    ("Orchard Harvest Ledger", [
        "Pickers at Hollowmere Orchard filled four hundred crates of Russet apples before the frost warning arrived.",
        ("The Bramley trees on the eastern slope gave a smaller crop because the blossom opened during a week of "
        "cold rain. Crates are weighed at the barn door and the weights are written in the green ledger kept "
        "beside the cider press."),
        "Windfall fruit goes to the juice line, and bruised pears are sold at half price at the Saturday farm stall.",
    ], ["Variety", "Crates", "Average weight (kg)"], [["Russet", "400", "17.5"], ["Bramley", "210", "18.2"],
                                                        ["Conference pear", "95", "16.0"]]),
    ("Bicycle Derailleur Adjustment Guide", [
        "Shift the chain onto the smallest rear sprocket before turning the high limit screw on the derailleur body.",
        ("If the chain rubs the spokes in the lowest gear, tighten the low limit screw a quarter turn at a time. "
        "Cable tension is set with the barrel adjuster, turned anticlockwise until the chain climbs cleanly to "
        "the next cog."),
        "A bent derailleur hanger causes noisy shifting that no amount of screw adjustment can cure.",
    ], ["Symptom", "Likely cause", "Adjustment"], [["Chain falls off", "High limit too loose", "H screw in"],
                                                     ["Slow upshift", "Cable too slack", "Barrel out"],
                                                     ["Rubbing spokes", "Low limit too loose", "L screw in"]]),
    ("Island Ferry Timetable Notes", [
        "The morning crossing from Port Ellery to Gannet Island leaves at a quarter past seven on weekdays.",
        ("Vehicles must queue in lane three no later than twenty minutes before departure or lose their reserved "
        "space. During the autumn gales the master may cancel sailings, and refunds are issued at the ticket "
        "kiosk on the quay."),
        "Bicycles travel free, but dogs must stay on a short lead on the open upper deck.",
    ], ["Departure", "Port Ellery", "Gannet Island"], [["Morning", "07:15", "08:05"], ["Midday", "12:40", "13:30"],
                                                        ["Evening", "18:20", "19:10"]]),
    ("Bakery Cost Sheet", [
        "The price of stoneground wholemeal flour from the Ashby mill went up by six percent this quarter.",
        ("Butter now makes up the largest share of the cost of each croissant, ahead of flour, sugar and labour. "
        "The sourdough loaf keeps its price because the starter culture needs only flour, water and a long cool "
        "rise."),
        "Leftover bread is turned into breadcrumbs or donated to the shelter on Mercer Lane each evening.",
    ], ["Item", "Ingredient cost", "Shop price"], [["Croissant", "0.48", "1.60"], ["Sourdough loaf", "0.95", "4.20"],
                                                     ["Rye roll", "0.22", "0.90"]]),
    ("Weather Station Monthly Summary", [
        "The hilltop weather station at Corrie Fell recorded its wettest October since the gauge was installed.",
        ("A south-westerly gust of ninety kilometres per hour bent the anemometer mast on the night of the "
        "twelfth. Overnight frost was logged on six mornings, mostly in the sheltered hollow beside the "
        "reservoir dam."),
        "The volunteer observers replaced the cracked thermometer screen with a louvred box made of painted cedar.",
    ], ["Measure", "October", "Ten-year mean"], [["Rainfall (mm)", "212", "131"], ["Frost mornings", "6", "4"],
                                                    ["Peak gust (km/h)", "90", "74"]]),
    ("Library Acquisitions Report", [
        ("The Elm Street branch library bought two hundred new picture books for the reading corner of the "
        "children's room."),
        ("Large-print editions of detective stories remain the most borrowed titles among readers over seventy. "
        "A donation of maritime charts from a retired harbour pilot will be catalogued in the local history room."),
        "Overdue fines were abolished in March, and the number of returned books has grown every month since.",
    ], ["Collection", "Bought", "Donated"], [["Picture books", "200", "35"], ["Large print", "60", "12"],
                                              ["Maritime charts", "0", "48"]]),
    ("Beekeeping Inspection Notes", [
        "Hive number four at the Larkspur apiary showed a laying queen and capped brood on seven frames.",
        ("Varroa mite counts on the sticky board were low, so no treatment is planned before the August "
        "inspection. The beekeeper added a second honey super because the colony was building wax comb along "
        "the crown board."),
        "Lime blossom nectar gives this summer's honey a pale colour and a faint taste of mint.",
    ], ["Hive", "Brood frames", "Mites per day"], [["Hive 2", "5", "1.5"], ["Hive 4", "7", "0.8"],
                                                    ["Hive 6", "4", "2.1"]]),
    ("Rooftop Solar Installation Checklist", [
        "Confirm that the roof rafters can carry the extra load of the mounting rails and the glass panels.",
        ("The inverter should be fitted in a shaded, ventilated place such as the garage wall, away from the "
        "boiler flue. Isolator switches are required on both the direct current side and the alternating "
        "current side of the inverter."),
        ("After commissioning, the installer photographs the meter reading and leaves the wiring diagram in the "
        "fuse cupboard."),
    ], ["Step", "Checked by", "Done"], [["Rafter survey", "Structural engineer", "yes"],
                                         ["Isolators fitted", "Electrician", "yes"],
                                         ["Meter photographed", "Installer", "no"]]),
    ("Museum Visitor Survey", [
        "Visitors to the Clockmakers' Museum rated the restored astronomical clock as the highlight of the tour.",
        ("Many families asked for more seating in the long gallery, where the pendulum regulators are displayed. "
        "The audio guide in Portuguese was used far more often than the curators expected."),
        ("Several visitors suggested opening the workshop on Sundays so that children can watch a pocket watch "
        "being repaired."),
    ], ["Question", "Agree", "Disagree"], [["The clock was the highlight", "81%", "6%"],
                                            ["The gallery needs seats", "64%", "11%"],
                                            ["Open the workshop on Sundays", "58%", "9%"]]),
    ("Railway Signal Maintenance Log", [
        "The semaphore signal at Brackley Junction was relit after the paraffin lamp failed during the night shift.",
        ("Point rodding between the signal box and the goods loop was greased and the compensator was reset. A "
        "cracked lever frame in the box will be replaced during the engineering possession at Easter."),
        "Track circuits on the down main line were tested and every relay dropped within the permitted time.",
    ], ["Asset", "Fault", "Status"], [["Signal BJ12", "Lamp out", "Relit"], ["Points 7A", "Stiff rodding", "Greased"],
                                       ["Lever frame", "Cracked", "Replace at Easter"]]),
    ("School Science Fair Rules", [
        "Every project at the Westbrook science fair must include a written hypothesis and a labelled diagram.",
        ("Experiments involving open flames, live animals or mains electricity need a teacher's signature in "
        "advance. Judges award marks for careful measurement, honest reporting of failures and a clear spoken "
        "explanation."),
        "The winning team will present its riverbank erosion model at the county exhibition in the town hall.",
    ], ["Criterion", "Marks", "Weight"], [["Hypothesis", "10", "20%"], ["Measurement", "20", "40%"],
                                           ["Explanation", "20", "40%"]]),
    ("Coffee Roasting Profile", [
        "The washed Ethiopian beans were charged into the drum roaster at two hundred degrees Celsius.",
        ("Yellowing began after four minutes, and the crack was heard shortly after the eight-minute mark. The "
        "roast was dropped into the cooling tray when the beans reached a light cinnamon colour."),
        "Tasters noted bergamot, apricot and black tea in the cup after a resting period of three days.",
    ], ["Stage", "Time (min)", "Bean temperature (C)"], [["Charge", "0:00", "200"], ["Yellowing", "4:00", "155"],
                                                           ["Drop", "9:30", "205"]]),
    ("Warehouse Inventory Audit", [
        "Auditors counted every pallet in aisle seven of the Riverside distribution warehouse on Tuesday night.",
        ("Fourteen cartons of ceramic floor tiles were found on the wrong racking level and moved to bay twelve. "
        "The barcode scanners disagreed with the stock system on three lines of garden hose fittings."),
        "Forklift drivers were reminded that the mezzanine floor has a load limit painted beside the stairs.",
    ], ["Line", "System count", "Shelf count"], [["Floor tiles", "120", "120"], ["Hose connector", "340", "322"],
                                                   ["Hose reel", "45", "47"]]),
    ("Mountain Trail Conditions", [
        ("The ridge path above Ravenscar Tarn is closed after a rockfall buried the stepping stones near the "
        "summit cairn."),
        ("Walkers heading to the waterfall should use the lower forest track, which is waymarked with yellow "
        "arrows. Snow lingers in the north-facing gully, and crampons are advised for the steep section below "
        "the col."),
        "The mountain rescue team asks hikers to leave their route and return time with the youth hostel warden.",
    ], ["Route", "Condition", "Advice"], [["Ridge path", "Closed", "Use the forest track"],
                                           ["Forest track", "Open", "Yellow arrows"],
                                           ["North gully", "Snow", "Crampons"]]),
    ("Chess Club Tournament Results", [
        "Margaret Okafor won the Ridgeway Chess Club autumn tournament with six wins and one draw.",
        ("The longest game of the event lasted ninety-one moves and ended in a rook and pawn endgame. Junior "
        "members played a separate section, which was won by a ten-year-old who favours the Sicilian Defence."),
        "Club nights move to the library annexe in winter because the church hall heating is being repaired.",
    ], ["Player", "Points", "Place"], [["M. Okafor", "6.5", "Winner"], ["D. Lindqvist", "5.0", "Runner-up"],
                                        ["P. Ahmed", "4.5", "Third"]]),
    ("Pottery Kiln Firing Schedule", [
        "The electric kiln at the Saltmarsh pottery studio is loaded with bisque ware every Thursday afternoon.",
        ("Stoneware glazes are fired to cone six, and the kiln is left to cool for a whole day before it is "
        "opened. Members must label each pot with their initials in iron oxide so that pieces are not mixed up."),
        "A shelf cracked during the last glaze firing, so wadding must now be placed under every lidded jar.",
    ], ["Firing", "Day", "Top temperature (C)"], [["Bisque", "Thursday", "1000"], ["Stoneware glaze", "Saturday", "1220"],
                                                    ["Raku", "Monthly", "950"]]),
    ("Community Garden Plot Allocation", [
        "Twenty-four raised beds at the Foundry Lane community garden will be allocated by ballot in February.",
        ("Applicants who live within walking distance of the garden gate receive two tickets in the ballot. The "
        "tool shed key is kept in a lockbox, and the code changes at the start of each growing season."),
        "Hosepipes may not be used during a drought order; gardeners should use the rainwater butts instead.",
    ], ["Bed size", "Beds", "Yearly fee"], [["Small", "10", "12"], ["Medium", "10", "18"], ["Large", "4", "25"]]),
    ("Lighthouse Keeper's Log", [
        "The keeper at Skerry Point lighthouse wound the clockwork rotation mechanism at dusk and again at midnight.",
        ("A fog bank rolled in from the west, so the foghorn sounded every thirty seconds until dawn. Two grey "
        "seals were seen hauled out on the flat rocks below the lantern gallery."),
        "The relief boat brought paraffin, fresh water and a crate of tinned peaches on Saturday.",
    ], ["Time", "Visibility", "Foghorn"], [["Dusk", "Good", "Off"], ["Midnight", "Poor", "On"],
                                            ["Dawn", "Moderate", "Off"]]),
    ("Natural Dye Recipes for Wool", [
        "Onion skins simmered for an hour give a warm golden yellow on wool mordanted with alum.",
        ("Madder root produces brick red in soft water, but the colour turns brown if the dye bath boils. Indigo "
        "vats need a reducing agent and should be stirred gently to avoid adding oxygen."),
        "Rinse the dyed skeins in water of the same temperature to stop the fibres from felting.",
    ], ["Dye", "Colour", "Mordant"], [["Onion skins", "Golden yellow", "Alum"], ["Madder root", "Brick red", "Alum"],
                                       ["Indigo", "Blue", "None"]]),
]


def page_text(index: int) -> str:
    """The words of one page as it is written, for ``--check``."""
    title, paragraphs, header, rows = PAGES[index]
    return " ".join([title, *paragraphs, *header, *(cell for row in rows for cell in row)])


def write(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2.4 * cm, bottomMargin=2.4 * cm, invariant=1,
                            title="OptiVision easy mode sample document", author="OptiVision RAG authors",
                            subject="20 short documents written for optivision-rag; MIT licence",
                            creator="scripts/make_sample_pdf.py")
    story = []
    for i, (title, paragraphs, header, rows) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Paragraph(f"Sample document {i + 1} of {len(PAGES)}", styles["Italic"]))
        story.append(Spacer(1, 0.4 * cm))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 0.25 * cm))
        table = Table([header, *rows], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(table)
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower().replace("’", "'"))


def check(path: Path) -> list[str]:
    """Pages whose extracted words differ from the script's, as messages (empty when equal)."""
    import pypdfium2 as pdfium

    problems = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        if len(pdf) != len(PAGES):
            return [f"{path.name} has {len(pdf)} pages, the script writes {len(PAGES)}"]
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                extracted = textpage.get_text_range()
            finally:
                textpage.close()
                page.close()
            want = _words(page_text(i))
            got = _words(extracted)
            if [w for w in got if w not in ("sample", "document", "of")] != \
                    [w for w in want if w not in ("sample", "document", "of")]:
                problems.append(f"page {i + 1}: the words differ from the script")
    finally:
        pdf.close()
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the file's text with the script instead of writing")
    a = ap.parse_args()
    if a.check:
        problems = check(OUT) if OUT.exists() else [f"{OUT.relative_to(ROOT)} is missing"]
        for problem in problems:
            print(problem)
        if problems:
            print("run scripts/make_sample_pdf.py")
            return 1
        print("sample.pdf matches the script")
        return 0
    write(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(PAGES)} pages, {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
