"""Write the briefs that synthetic conversations are generated from.

    python -m bench.embedder.briefs --count 500 --batches 25 --out briefs/

Each brief fixes who is talking, how they know each other, what happens
in their lives over the period, and how they write, so that a batch of
generated conversations does not come out as five hundred versions of
the same one. Everything is drawn from the lists below with a seeded
generator: the same arguments give the same briefs.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

NAMES = ["Aiko", "Amara", "Andre", "Anya", "Arjun", "Beatriz", "Callum", "Carmen", "Chidi", "Dara", "Dmitri", "Elena", "Emeka", "Farah", "Finn", "Gabriela", "Hana", "Hassan", "Imani", "Ingrid", "Ivan", "Jamal", "Jun", "Kavya", "Keanu", "Laila", "Lars", "Leilani", "Luca", "Mateo", "Mei", "Miriam", "Nadia", "Naveen", "Niamh", "Omar", "Paloma", "Priya", "Quentin", "Rafael", "Rosa", "Sanjay", "Sefa", "Siobhan", "Tariq", "Tess", "Thabo", "Uma", "Viktor", "Wren", "Xavier", "Yara", "Yusuf", "Zofia", "Ben", "Claire", "Diego", "Esther", "Greg", "Holly", "Isaac", "Joan", "Kurt", "Lena", "Marcus", "Nora", "Otis", "Pearl", "Ruth", "Stan", "Tina", "Walt"]

JOBS = """nurse|welder|primary school teacher|freelance illustrator|line cook|data analyst|
bus driver|veterinary technician|graduate student in geology|retired postal worker|barista|
civil engineer|hairdresser|paramedic|bookkeeper|software tester|farmer|librarian|
real estate agent|electrician|physiotherapist|warehouse supervisor|journalist|
stay-at-home parent|piano tutor|park ranger|pharmacist|call-centre agent|carpenter|
flight attendant|social worker|dental hygienist|bike mechanic|museum guide|
university admissions officer|landscape gardener|translator|baker|plumber|
high school student|undergraduate in nursing|unemployed and job hunting""".replace("\n", "").split("|")

RELATIONSHIPS = """old college roommates|siblings living in different cities|coworkers on the same shift|
neighbours in the same building|parent and adult child|members of the same running club|
friends who met in an online game|cousins|a mentor and a former intern|former bandmates|
friends from a language class|two parents whose kids are in the same class|
long-distance best friends|housemates|friends from a support group|pen pals who have never met|
gym partners|an aunt or uncle and their niece or nephew|friends who volunteer together|
former colleagues who kept in touch""".replace("\n", "").split("|")

EVENTS = """adopts a pet and deals with a health scare for it|changes jobs after a difficult interview process|
trains for a first race and gets injured part-way|moves to a new flat and has trouble with the landlord|
plans a trip abroad, with a booking that goes wrong|starts an evening class and makes something specific|
cares for a parent after surgery|has a baby in the family, with a naming discussion|
buys a used car that needs repairs|takes up a new hobby and buys specific gear|
goes through a breakup and later starts dating again|organises a surprise party with a guest list and a venue|
renovates a kitchen, with costs and delays|applies to a degree programme and waits for a decision|
starts a small side business and gets a first customer|is diagnosed with a food allergy and changes diet|
joins a band or choir and prepares for a performance|has a falling-out with a friend and patches it up|
loses and replaces an important item|learns to cook a family recipe from a relative|
fosters a child or hosts an exchange student|gets a promotion that comes with a relocation offer|
takes a long hiking or cycling trip with named stops|deals with a flood, a break-in or a fire at home|
studies for a professional exam and gets the result|plants a garden and tracks what grows|
volunteers at a shelter or food bank and meets someone memorable|buys a first home, with an offer that falls through|
recovers from an operation with a rehab schedule|reconnects with an estranged relative|
goes to a wedding and has a role in it|starts therapy or a new medication and reports how it goes|
teaches a child or a parent a new skill|enters a competition and reports the placing|
saves for something specific and reaches or misses the goal|gets a tattoo, a piercing or a drastic haircut with a story behind it|
switches to a new sport and joins a named team|has a visa or immigration matter with dates|
reads a book or watches a series the other recommended and reacts|hosts relatives for a holiday, with a menu and a mishap""".replace("\n", "").split("|")

STYLES = """short lower-case texts with little punctuation|full, careful sentences|lots of jokes and teasing|
warm and a bit long-winded|dry and understated|enthusiastic, with exclamation marks|
often replies late and apologises for it|asks many follow-up questions|
mixes in the odd word of another language they speak|practical and to the point|
anxious, and tends to over-explain|nostalgic, and often brings up shared history""".replace("\n", "").split("|")

PLACES = """Leeds|Tucson|Kraków|Durban|Osaka|Recife|Winnipeg|Auckland|Cork|Pune|Tampa|Lyon|Accra|
Hobart|Bergen|Cebu|Guadalajara|Izmir|Halifax|Perth|a small town in Ohio|a village in Wales|
rural Saskatchewan|the suburbs of Atlanta|a farm outside Nairobi|a coastal town in Portugal""".replace("\n", "").split("|")


def brief(index: int, rng: random.Random) -> dict:
    first, second = rng.sample(NAMES, 2)
    sessions = rng.randint(5, 9)
    start = date(2021, 1, 1) + timedelta(days=rng.randrange(0, 4 * 365))
    dates, day = [], start
    for _ in range(sessions):
        dates.append(day.isoformat())
        day += timedelta(days=rng.choice([1, 2, 4, 7, 10, 14, 21, 30, 45, 60]))
    people = []
    for name in (first, second):
        people.append({
            "name": name,
            "age": rng.randint(17, 74),
            "job": rng.choice(JOBS),
            "lives_in": rng.choice(PLACES),
            "writes": rng.choice(STYLES),
            "over_this_period": rng.sample(EVENTS, 2),
        })
    return {
        "id": f"c{index:04d}",
        "relationship": rng.choice(RELATIONSHIPS),
        "people": people,
        "session_dates": dates,
        "turns_per_session": rng.randint(10, 18),
        "questions": 20,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    briefs = [brief(i, rng) for i in range(args.count)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = -(-args.count // args.batches)
    for batch in range(args.batches):
        part = briefs[batch * size:(batch + 1) * size]
        (out / f"batch{batch:02d}.json").write_text(json.dumps(part, indent=1) + "\n")
    print(f"{len(briefs)} briefs in {args.batches} files under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
