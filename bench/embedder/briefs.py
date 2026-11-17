"""Write the briefs that synthetic conversations are generated from.

    python -m bench.embedder.briefs --count 500 --batches 25 --out briefs/
    python -m bench.embedder.briefs --kind assistant --start 500 --count 260 --batches 13 --out briefs/
    python -m bench.embedder.briefs --kind notes --start 760 --count 240 --batches 12 --out briefs/

Three kinds: ``chat`` is two people messaging each other, ``assistant``
is one person talking to an AI assistant over weeks, and ``notes`` is
the memories a working agent saves about a project and its owner.

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


ASSISTANT_USES = """meal planning around a health condition|learning a language for a move abroad|
training for an endurance event|a job search with several applications in flight|
planning a wedding on a budget|managing a chronic illness and its medications|
writing a novel, with characters and plot decisions|renovating a house room by room|
a first year of running a small shop|caring for an ageing parent from a distance|
studying for a professional licence|planning a long trip with several stops|
raising a puppy or a kitten|a personal finance overhaul with specific accounts and goals|
building a vegetable garden through one season|a career change into a new field|
home-schooling a child for a term|organising a community event|
restoring an old car or motorbike|getting a diagnosis and comparing treatment options|
learning an instrument as an adult|moving to a new city and settling in|
a research project for a degree|setting up a home network and smart devices|
planning for a baby|a fitness and weight plan with numbers that change""".replace("\n", "").split("|")

PROJECTS = """a mobile app for a dog-walking business|the billing service of a small SaaS company|
an online shop for handmade ceramics|a data pipeline for a regional weather service|
a booking system for a dental practice|an internal tool for tracking warehouse stock|
a marketing site and blog for a law firm|a multiplayer browser game|
a machine-learning model that sorts support tickets|a payroll export for a restaurant group|
the firmware and dashboard for a greenhouse sensor|a migration from one database to another|
a customer newsletter and its automation|a quarterly financial report and its spreadsheets|
a research paper with experiments and reviews|a podcast with a release schedule and guests|
an event website with ticketing|a property-management back office|
a public API with rate limits and client libraries|a school timetable generator""".replace("\n", "").split("|")

AGENTS = ["claude-code", "codex", "gemini", "cursor", "assistant", "ops-bot", "research-bot", "inbox-bot"]

NOTE_TOPICS = """decisions and why they were made|the owner's preferences about style and tools|
names of people, vendors and customers and what each is like|dates, deadlines and what moved|
things that went wrong and how they were fixed|numbers: prices, limits, versions, sizes|
where things live: files, accounts, URLs, folders (invented)|
things the owner said never to do|open questions and who is waiting on whom|
facts about the owner's life that affect the work""".replace("\n", "").split("|")


def _dates(rng: random.Random, sessions: int, gaps: list[int]) -> list[str]:
    day = date(2022, 1, 1) + timedelta(days=rng.randrange(0, 4 * 365))
    out = []
    for _ in range(sessions):
        out.append(day.isoformat())
        day += timedelta(days=rng.choice(gaps))
    return out


def assistant_brief(index: int, rng: random.Random) -> dict:
    return {
        "id": f"c{index:04d}",
        "kind": "assistant",
        "people": [{"name": "user"}, {"name": "assistant"}],
        "user": {
            "first_name": rng.choice(NAMES),
            "age": rng.randint(19, 74),
            "job": rng.choice(JOBS),
            "lives_in": rng.choice(PLACES),
            "writes": rng.choice(STYLES),
        },
        "uses_the_assistant_for": rng.sample(ASSISTANT_USES, 3),
        "session_dates": _dates(rng, rng.randint(6, 10), [1, 2, 3, 5, 7, 10, 14, 21, 30]),
        "turns_per_session": rng.choice([6, 8, 8, 10, 12]),
        "questions": 20,
    }


def notes_brief(index: int, rng: random.Random) -> dict:
    agents = rng.sample(AGENTS, rng.choice([1, 2, 2, 3]))
    return {
        "id": f"c{index:04d}",
        "kind": "notes",
        "people": [{"name": name} for name in agents],
        "owner": {
            "first_name": rng.choice(NAMES),
            "job": rng.choice(JOBS),
            "lives_in": rng.choice(PLACES),
        },
        "projects": rng.sample(PROJECTS, 2),
        "notes_should_cover": rng.sample(NOTE_TOPICS, 6),
        "session_dates": _dates(rng, rng.randint(6, 10), [1, 1, 2, 3, 5, 7, 10, 14]),
        "turns_per_session": rng.randint(8, 14),
        "questions": 20,
    }


KINDS = {"chat": brief, "assistant": assistant_brief, "notes": notes_brief}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=sorted(KINDS), default="chat")
    parser.add_argument("--start", type=int, default=0, help="number of the first brief")
    parser.add_argument("--first-batch", type=int, default=0, help="number of the first file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rng = random.Random(f"{args.kind}-{args.seed}" if args.kind != "chat" else args.seed)
    briefs = [KINDS[args.kind](args.start + i, rng) for i in range(args.count)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = -(-args.count // args.batches)
    for batch in range(args.batches):
        part = briefs[batch * size:(batch + 1) * size]
        (out / f"batch{args.first_batch + batch:02d}.json").write_text(json.dumps(part, indent=1) + "\n")
    print(f"{len(briefs)} briefs in {args.batches} files under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
