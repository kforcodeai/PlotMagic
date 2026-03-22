#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CHAPTER_DIR = Path("data/kerala/kmbr_muncipal_rules_md")
COMBINED_SOURCE = Path("evaluation/kmbr/kmbr_municipality_rules_combined.md")
OUT_JSONL = Path("evaluation/kmbr/kmbr_multihop_retrieval_dataset.jsonl")
OUT_JSON = Path("evaluation/kmbr/kmbr_multihop_retrieval_dataset.json")

RULE_HEADER_RE = re.compile(r"^##\s+(\d+[A-Za-z]?)\.")


# Each evidence item uses a rule-scoped literal substring start/end match.
# Matching is case-insensitive and searches within the selected rule block.
ENTRIES: list[dict[str, Any]] = [
    {
        "id": "kmbr_mh_001",
        "query": "For a land development proposal within 100m of defence property and 30m of railway boundary, what consultations are mandatory and what interim-response rule applies if final remarks are delayed?",
        "evidence": [
            {
                "rule": "5",
                "start": "within a distance of 100 meters from any property maintained by",
                "end": "before issuing permit.",
            },
            {
                "rule": "5",
                "start": "land within 30 meters",
                "end": "before issuing permit.",
                "start_occurrence": 1,
            },
            {
                "rule": "5",
                "start": "In cases where final remarks are not received within the 30 days time",
                "end": "received from the concerned Defense/Railway Authority.",
            },
        ],
    },
    {
        "id": "kmbr_mh_002",
        "query": "For a building permit case near both defence land and railway boundary, what are the consultation distances, response windows, and the secretary's treatment of delayed final remarks?",
        "evidence": [
            {
                "rule": "7",
                "start": "within a distance of 100 meters from any",
                "end": "before issuing permit.",
            },
            {
                "rule": "7",
                "start": "within 30 meters",
                "end": "before issuing permit.",
            },
            {
                "rule": "7",
                "start": "In cases where final remarks are not received within the 30 days time",
                "end": "if any interim reply is received from the Defense/Railway Authority.",
            },
        ],
    },
    {
        "id": "kmbr_mh_003",
        "query": "If an owner claims permit exemption for interior work and shifting building location, what advance intimation and completion-plan conditions apply, and when can minor deviations proceed without revised permit?",
        "evidence": [
            {
                "rule": "10",
                "start": "no building permit",
                "end": "interior decoration without any structural alterations ;",
            },
            {
                "rule": "10",
                "start": "under item (ix) shall be",
                "end": "communicated to the applicant within ten days.",
            },
            {
                "rule": "10",
                "start": "Provided further that the changing of the location under item (x)",
                "end": "completion plan.",
            },
            {
                "rule": "17",
                "start": "no permit is necessary for effecting minor deviations",
                "end": "submitted along with completion certificate.",
            },
        ],
    },
    {
        "id": "kmbr_mh_004",
        "query": "When both the Secretary and then the Council delay permit decisions, when does deemed permission arise, how is execution treated, and how does regularization chapter treat such statutory-delay cases?",
        "evidence": [
            {
                "rule": "15",
                "start": "if the Secretary, neither approves nor",
                "end": "approval or permission should be given or not.",
            },
            {
                "rule": "15",
                "start": "Where the Council does not, within one month",
                "end": "bye-laws made thereunder.",
            },
            {
                "rule": "15",
                "start": "Provided that such execution of work shall be considered as duly",
                "end": "if it otherwise complies with rule provisions.",
            },
            {
                "rule": "143",
                "start": "after the statutory period specified in rules 15,",
                "end": "not one requiring regularization.",
            },
        ],
    },
    {
        "id": "kmbr_mh_005",
        "query": "What is the permit validity lifecycle for development/building permits, including extension count, extension and renewal fee rates, and the hard cap for older permits?",
        "evidence": [
            {
                "rule": "15A",
                "start": "shall be valid for three years from the date of issue.",
                "end": "grant extension twice, for further periods of three",
            },
            {
                "rule": "15A",
                "start": "fee for extension of period of permits shall be ten %",
                "end": "force at the time of renewal.",
            },
            {
                "rule": "15A",
                "start": "In case the period of validity",
                "end": "shall not exceed nine years.",
            },
        ],
    },
    {
        "id": "kmbr_mh_006",
        "query": "On what grounds can a permit be suspended or revoked, and what pre-revocation procedural protection must be given to the permit holder?",
        "evidence": [
            {
                "rule": "16",
                "start": "The Secretary shall suspend or revoke any permit issued under",
                "end": "threat to life or property:",
            },
            {
                "rule": "16",
                "start": "Provided that before revoking permit",
                "end": "duly considered by the Secretary.",
            },
        ],
    },
    {
        "id": "kmbr_mh_007",
        "query": "For unauthorized deviations during construction, what are the secretary's alteration powers, show-cause sequence, and when can deviated works avoid demolition through regularization compliance?",
        "evidence": [
            {
                "rule": "18",
                "start": "that the construction, reconstruction or alteration of any building",
                "end": "issued there under or any direction or requisition lawfully given",
            },
            {
                "rule": "18",
                "start": "he may make a provisional order requiring the owner",
                "end": "shall refrain from proceeding with the work.",
            },
            {
                "rule": "18",
                "start": "serve a copy of the provisional order",
                "end": "to show cause within",
            },
            {
                "rule": "18",
                "start": "Provided that any construction or reconstruction",
                "end": "in these rules.",
            },
        ],
    },
    {
        "id": "kmbr_mh_008",
        "query": "If illegal building work continues after notice, what immediate stoppage powers, police enforcement, and cost-recovery mechanism does the Secretary have?",
        "evidence": [
            {
                "rule": "19",
                "start": "has been commenced",
                "end": "to stop the same forthwith.",
            },
            {
                "rule": "19",
                "start": "any police officer to remove such person",
                "end": "requisition accordingly.",
            },
            {
                "rule": "19",
                "start": "After the requisition",
                "end": "recoverable from person as an arrear of property tax",
            },
        ],
    },
    {
        "id": "kmbr_mh_009",
        "query": "When a permitted plot is transferred before completion, what must transferor and transferee file, what fee/document package is needed, and what timeline binds the Secretary?",
        "evidence": [
            {
                "rule": "21",
                "start": "Every person holding",
                "end": "transfer or otherwise of the permit.",
            },
            {
                "rule": "21",
                "start": "whose favor any property is transferred",
                "end": "fee of Rs. 25.",
            },
            {
                "rule": "21",
                "start": "if convinced that the transfer will not in anyway badly affect",
                "end": "15 days from the date of receipt of the request.",
            },
        ],
    },
    {
        "id": "kmbr_mh_010",
        "query": "After project completion, what signing requirements apply to completion certificates, what are the statutory timelines for development/occupancy certificates, and when are they deemed issued?",
        "evidence": [
            {
                "rule": "22",
                "start": "submit a completion certificate certified",
                "end": "also as in Appendix F.",
            },
            {
                "rule": "22",
                "start": "issue a development certificate",
                "end": "certificate has been duly issued to him.",
            },
            {
                "rule": "22",
                "start": "occupancy certificate in the form in Appendix H",
                "end": "certificate has been duly issued to him.",
            },
            {
                "rule": "22",
                "start": "if he intends to occupy the building",
                "end": "completed part.",
            },
        ],
    },
    {
        "id": "kmbr_mh_011",
        "query": "For a plot that is flood-prone, in CRZ, and near overhead electric lines, what baseline prohibitions/restrictions apply and when can a single-storied exception proceed with external clearance?",
        "evidence": [
            {
                "rule": "23",
                "start": "on a plot liable to flood or on a slop forming an angle of more",
                "end": "services.",
            },
            {
                "rule": "23",
                "start": "area notified by the Government of India as a coastal regulation",
                "end": "as amended from time to time.",
            },
            {
                "rule": "23",
                "start": "any overhead electric supply line",
                "end": "before issue of permit",
            },
        ],
    },
    {
        "id": "kmbr_mh_012",
        "query": "What minimum front/rear/side/interior open-space rules apply up to 10m building height, and how do setbacks scale once height exceeds 10m?",
        "evidence": [
            {
                "rule": "24",
                "start": "minimum front yard of 3 meters",
                "end": "minimum of 1.00 meter",
            },
            {
                "rule": "24",
                "start": "side yard of 1.20 meters on one side and a minimum of 1.00 meter",
                "end": "voluntarily agrees for the same in writing.",
            },
            {
                "rule": "24",
                "start": "Any room intended",
                "end": "minimum width of 1.5 meters.",
            },
            {
                "rule": "24",
                "start": "For buildings above",
                "end": "at their level.",
            },
        ],
    },
    {
        "id": "kmbr_mh_013",
        "query": "How are street-centerline and boundary distances determined, what relaxations apply for cul-de-sacs/lanes, and what parallel planning restrictions still continue to apply?",
        "evidence": [
            {
                "rule": "25",
                "start": "minimum distance",
                "end": "shall be 3 meters.",
            },
            {
                "rule": "25",
                "start": "case of cul-de-sac of whatever width but not exdeeing 250metres",
                "end": "line of the road to the building.",
            },
            {
                "rule": "25",
                "start": "case of lanes not exceeding 75 meters",
                "end": "lane.",
            },
            {
                "rule": "25",
                "start": "Any restriction",
                "end": "sub rule (1).",
            },
        ],
    },
    {
        "id": "kmbr_mh_014",
        "query": "For construction abutting national/state/notified roads, what is the absolute no-build distance and what limited access/projection structures are still allowed within that belt?",
        "evidence": [
            {
                "rule": "26",
                "start": "No person shall construct any building other than compound",
                "end": "other  roads-notified by Municipality:",
            },
            {
                "rule": "26",
                "start": "Provided that open ramps or bridges or steps",
                "end": "into such 3 meters.",
            },
        ],
    },
    {
        "id": "kmbr_mh_015",
        "query": "In mixed-use classification, when does the most restrictive occupancy govern, and how do A1 and A2 thresholds distinguish small professional/residential uses from special residential uses?",
        "evidence": [
            {
                "rule": "30",
                "start": "Any building",
                "end": "under the most restrictive group.,",
            },
            {
                "rule": "30",
                "start": "professional offices or spaces for advocates",
                "end": "50 sq.m floor area",
            },
            {
                "rule": "30",
                "start": "hotels not exceeding 150 sq.m floor",
                "end": "area are included in this group.",
            },
            {
                "rule": "30",
                "start": "Special Residential building shall include all lodging or rooming",
                "end": "included in this group.",
            },
        ],
    },
    {
        "id": "kmbr_mh_016",
        "query": "How is FAR computed, what ceiling logic applies by occupancy, what extra payment unlocks higher FAR, and what range of occupancy-specific table values is codified?",
        "evidence": [
            {
                "rule": "31",
                "start": "Floor area ratio ie, F.A.R. shall be calculated as shown",
                "end": "Table 2 below.",
            },
            {
                "rule": "31",
                "start": "For permitting for FAR shown in column (5) of Table 2",
                "end": "permissible under column (4) shall be paid.",
            },
            {
                "rule": "31",
                "start": "| 1 | Residential <br> A1 | 65 | 3.00 | 4 |",
                "end": "| 12 | Hazardous <br> I(2) | 25 | 0.70 | 0 |",
            },
        ],
    },
    {
        "id": "kmbr_mh_017",
        "query": "For parking compliance, what minimum bay dimensions apply, how are A1/A2 parking rates and dining add-on computed, what two-wheeler surcharge is mandatory, and when do loading bays become compulsory?",
        "evidence": [
            {
                "rule": "34",
                "start": "parking motor cars shall",
                "end": "1.5 sq. mt. respectively.",
            },
            {
                "rule": "34",
                "start": "Group A1- Residential",
                "end": "(d) Single unit (exceeding 200 sq.m of carpet area)",
            },
            {
                "rule": "34",
                "start": "At the rate of one parking space for every 30 sq.m",
                "end": "Buildings attached with eating facility.",
            },
            {
                "rule": "34",
                "start": "25% of that area shall be",
                "end": "parking scooters or cycles.",
            },
            {
                "rule": "34",
                "start": "in the case of Group F Mercantile or",
                "end": "floor area.",
            },
        ],
    },
    {
        "id": "kmbr_mh_018",
        "query": "For a multi-storey project, what staircase minimums and external-stair conditions apply generally, and what additional fire-escape trigger and dimensional requirements apply by occupancy/storey?",
        "evidence": [
            {
                "rule": "39",
                "start": "having more than four floors including basement or sunken floors",
                "end": "shall have at least two staircases",
            },
            {
                "rule": "39",
                "start": "Provided that when",
                "end": "not be provided.",
            },
            {
                "rule": "42",
                "start": "residential",
                "end": "exceeding two storeys above ground level.",
            },
            {
                "rule": "42",
                "start": "width of fire",
                "end": "shall have a straight flight.",
            },
        ],
    },
    {
        "id": "kmbr_mh_019",
        "query": "When do lift obligations trigger under general rules for hospitals vs other occupancies, and how do high-rise apartment thresholds add stretcher-carrying requirements?",
        "evidence": [
            {
                "rule": "48",
                "start": "exceeding 3 storeys (excluding sunken floors) in the case of",
                "end": "required number of staircases as per rule 39.",
            },
            {
                "rule": "48",
                "start": "Whenever more than one lift is",
                "end": "stretcher",
            },
            {
                "rule": "118",
                "start": "having more than 16 dwelling",
                "end": "shall be one capable of carrying a stretcher.",
            },
        ],
    },
    {
        "id": "kmbr_mh_020",
        "query": "What lighting/ventilation baseline rules govern habitable rooms and deep-room lighting limits, and what bathroom/latrine air-shaft dimensions scale by number of storeys?",
        "evidence": [
            {
                "rule": "49",
                "start": "Every habitable",
                "end": "by artificial means.",
            },
            {
                "rule": "49",
                "start": "No portion of a room shall be assumed to be lighted if it is more",
                "end": "unless it is artificially lighted",
            },
            {
                "rule": "49",
                "start": "Every bathroom or",
                "end": "Table 8.",
            },
            {
                "rule": "49",
                "start": "| 1. | Upto 3 | 1.08 | 0.9 |",
                "end": "| 4. | Above 10 | 5.0 | 2.0 |",
            },
        ],
    },
    {
        "id": "kmbr_mh_021",
        "query": "For large residential apartment projects, when is recreation space mandatory, how is its minimum area calculated/located, and which small family-residential buildings are exempt from most Chapter VI provisions?",
        "evidence": [
            {
                "rule": "50",
                "start": "Any residential",
                "end": "provided with a recreation space of suitable size.",
            },
            {
                "rule": "50",
                "start": "shall have not less than 7.5 % of the",
                "end": "other utility areas.",
            },
            {
                "rule": "51",
                "start": "Family residential",
                "end": "rule 42 and rule 44",
            },
        ],
    },
    {
        "id": "kmbr_mh_022",
        "query": "Under residential occupancy rules, what doorway redundancy threshold applies, what hazardous-use restrictions/exceptions exist, and when is Fire Force approval mandatory?",
        "evidence": [
            {
                "rule": "53",
                "start": "exceeding",
                "end": "leading to separate exit",
            },
            {
                "rule": "53",
                "start": "No hazardous use",
                "end": "poultry, dairy or kennel in residential",
            },
            {
                "rule": "53",
                "start": "buildings exceeding three storeys above ground level",
                "end": "before issue of the building permit.",
            },
            {
                "rule": "53",
                "start": "All other requirements",
                "end": "amendment No. 3, part IV.",
            },
        ],
    },
    {
        "id": "kmbr_mh_023",
        "query": "For educational, medical/hospital, and office/business occupancies, what DTP/CTP approval thresholds apply, what open-yard minima are mandated, and what extra hospital/fire requirements are triggered?",
        "evidence": [
            {
                "rule": "54",
                "start": "Approval of the District Town Planner shall be obtained",
                "end": "with more than 500 sq.m of area;",
            },
            {
                "rule": "54",
                "start": "All buildings upto 10 metres height under educational, medical/",
                "end": "three metres exceeding that height:",
            },
            {
                "rule": "54",
                "start": "where the height of the building exceeds 10 rnetres",
                "end": "50 cms for every 3 metres increase in height.]",
            },
            {
                "rule": "54",
                "start": "Every hospital",
                "end": "and pathogenic wastes.",
            },
            {
                "rule": "54",
                "start": "buildings exceeding three floors from ground level",
                "end": "before issuing permit.",
            },
        ],
    },
    {
        "id": "kmbr_mh_024",
        "query": "For industrial/small-industrial projects, what DTP/CTP thresholds, open-yard matrix, access-width rules, and pollution/fire approval obligations must be satisfied?",
        "evidence": [
            {
                "rule": "57",
                "start": "Approval of district town planner shall be obtained for the usage",
                "end": "with more than 500 sq.m",
            },
            {
                "rule": "57",
                "start": "open spaces not less than that specified",
                "end": "3.0",
            },
            {
                "rule": "57",
                "start": "industrial growth centres or industrial development",
                "end": "3 metres in all sides",
            },
            {
                "rule": "57",
                "start": "width of every",
                "end": "not less than 3 metres.",
            },
            {
                "rule": "57",
                "start": "approval of the Pollution Control Board shall",
                "end": "be obtained in all cases.",
            },
            {
                "rule": "57",
                "start": "certificate of approval from the Director",
                "end": "issuing building permit-",
            },
        ],
    },
    {
        "id": "kmbr_mh_025",
        "query": "For storage/warehousing buildings above threshold size, what planning approvals, open-yard standards, rat-proofing requirements, and fire approvals are mandatory?",
        "evidence": [
            {
                "rule": "58",
                "start": "with area exceeding 300 sq.m.",
                "end": "with area exceeding 300 sq.m.",
            },
            {
                "rule": "58",
                "start": "Approval of District Town Planner shall be obtained for the usage",
                "end": "exceeding 500 sq.m",
            },
            {
                "rule": "58",
                "start": "minimum open yards",
                "end": "Average 3 metres with minimum 1.5 metres",
            },
            {
                "rule": "58",
                "start": "all openings",
                "end": "rat-proof-material.",
            },
            {
                "rule": "58",
                "start": "certificate of approval from the Director",
                "end": "issuing building permit;",
            },
        ],
    },
    {
        "id": "kmbr_mh_026",
        "query": "Under hazardous occupancy controls, what DTP/CTP thresholds and all-round open-yard minima apply by subgroup, and what additional fire/petrol/crematorium distance requirements must be checked?",
        "evidence": [
            {
                "rule": "59",
                "start": "Approval of District Town Planner shall be obtained for the usage",
                "end": "lay out of buildings exceeding 500 sq.m areas.",
            },
            {
                "rule": "59",
                "start": "minimum width",
                "end": "Group 1 (2) hazardous occupancy.",
            },
            {
                "rule": "59",
                "start": "certificate of approval from the Director of Fire Force",
                "end": "before issuing building permi",
            },
            {
                "rule": "59",
                "start": "retail dispensing",
                "end": "from any point of the marked boundary of its premises.",
            },
            {
                "rule": "59",
                "start": "kiosk or sales office shall have a minimum open space of 1.00",
                "end": "other than that abutting the street.",
            },
            {
                "rule": "59",
                "start": "minimum",
                "end": "open space all round the crematorium.",
                "start_occurrence": 2,
            },
        ],
    },
    {
        "id": "kmbr_mh_027",
        "query": "For small-plot construction, what eligibility and anti-fragmentation rules apply, what floor/setback constraints apply, which major provisions are waived, and which procedural chapter still governs applications?",
        "evidence": [
            {
                "rule": "60",
                "start": "plots not exceeding 125 sq.m of area",
                "end": "proposed plot.",
            },
            {
                "rule": "61",
                "start": "number of floors allowed shall be",
                "end": "three",
            },
            {
                "rule": "62",
                "start": "minimum distance between the plot boundary abutting any",
                "end": "shall be 2",
            },
            {
                "rule": "62",
                "start": "front",
                "end": "1.20 metres",
                "start_occurrence": 1,
            },
            {
                "rule": "62",
                "start": "Any one side",
                "end": "agree",
            },
            {
                "rule": "63",
                "start": "Provisions regarding FAR, coverage",
                "end": "under this Chapter.",
            },
            {
                "rule": "64",
                "start": "shall be as described in",
                "end": "Schedule II respectively.",
            },
        ],
    },
    {
        "id": "kmbr_mh_028",
        "query": "What are the core controls for new row buildings: where they are allowed, unit cap, per-unit plot size, floor limit, and the major provisions expressly disapplied?",
        "evidence": [
            {
                "rule": "65",
                "start": "shall permit the construction or",
                "end": "decided to allow row buildings.",
            },
            {
                "rule": "66",
                "start": "Number of dwelling units in a row of buildings shall not exceed",
                "end": "ten.",
            },
            {
                "rule": "67",
                "start": "one unit shall not exceed 85 sq.m.",
                "end": "one unit shall not exceed 85 sq.m.",
            },
            {
                "rule": "69",
                "start": "Maximum number of floors permitted shall be two and a staircase room.",
                "end": "Maximum number of floors permitted shall be two and a staircase room.",
            },
            {
                "rule": "70",
                "start": "Provisions regarding F.A.R., Coverage",
                "end": "be applicable to row buildings.",
            },
        ],
    },
    {
        "id": "kmbr_mh_029",
        "query": "For row-building cases, how are permit applications filed/processed and what broad reconstruction freedoms apply to pre-existing row buildings even outside declared row streets?",
        "evidence": [
            {
                "rule": "71",
                "start": "Application for",
                "end": "either jointly or individually.",
            },
            {
                "rule": "71",
                "start": "submission and disposal of application",
                "end": "as in Chapter II.",
            },
            {
                "rule": "72",
                "start": "reconstruction, repair, alternation",
                "end": "plot area, use and set backs provided.",
            },
        ],
    },
    {
        "id": "kmbr_mh_030",
        "query": "For buildings under approved welfare/government schemes, what sponsoring bodies are covered, what plinth/floor caps and setback norms apply, what provisions are waived, and how do layout/permit pathways differ for individuals vs agencies?",
        "evidence": [
            {
                "rule": "73",
                "start": "financed or built by Government, Municipality, Housing",
                "end": "Scheme for economically weaker sections.",
            },
            {
                "rule": "74",
                "start": "Total plinth area of the building shall not exceed 50 Sq. mts.",
                "end": "staircase room.",
            },
            {
                "rule": "75",
                "start": "minimum distance between the",
                "end": "shall be minimum 1.50 metres.",
            },
            {
                "rule": "76",
                "start": "Provisions regarding FAR, coverage",
                "end": "under this",
            },
            {
                "rule": "77",
                "start": "Layout approval shall be obtained",
                "end": "not required.",
            },
            {
                "rule": "78",
                "start": "application in",
                "end": "no building permit is necessary.",
            },
        ],
    },
    {
        "id": "kmbr_mh_031",
        "query": "In road-surrender cases, what project and surrender preconditions must be met, when is benefit denied or restored under modified schemes, and how is surrendered land custody/use controlled?",
        "evidence": [
            {
                "rule": "79",
                "start": "plots left after part of the same",
                "end": "modifications in this chapter",
            },
            {
                "rule": "79",
                "start": "benefit of the provisions in this chapter shall not be",
                "end": "implementation of the scheme in total.",
            },
            {
                "rule": "79",
                "start": "scheme is modified or the width of the road is reduced",
                "end": "reduced width in total.",
            },
            {
                "rule": "79",
                "start": "surrendering",
                "end": "rule 85",
            },
            {
                "rule": "79",
                "start": "surrendered land shall not be used for purposes other than",
                "end": "in the scheme",
            },
        ],
    },
    {
        "id": "kmbr_mh_032",
        "query": "For road-surrender plots, how can FAR/coverage treatment improve, how is surrendered land counted, and what front-distance reductions and general setback exemptions can Special Committee decisions unlock?",
        "evidence": [
            {
                "rule": "80",
                "start": "Maximum coverage",
                "end": "Group 1 (2)",
            },
            {
                "rule": "80",
                "start": "committee constituted under rule 85 may, taking into account the",
                "end": "without claiming any additional fee:",
            },
            {
                "rule": "80",
                "start": "area of the land surrendered shall also be taken into account",
                "end": "under this",
            },
            {
                "rule": "80",
                "start": "distance from the boundary of the proposed road or proposed",
                "end": "minimum of 1.50 metres distance",
            },
            {
                "rule": "80",
                "start": "General provision",
                "end": "to the constructions under this chapter.",
            },
        ],
    },
    {
        "id": "kmbr_mh_033",
        "query": "How does parking relief work under road-surrender provisions (including hard minimums), and what Special Committee structure/process controls the permit decisions?",
        "evidence": [
            {
                "rule": "84",
                "start": "minimum of fifty % of the parking",
                "end": "shall be provided.",
            },
            {
                "rule": "84",
                "start": "allow further reduction in the off street parking requirements",
                "end": "minimum of twenty-five % of the parking area required",
            },
            {
                "rule": "85",
                "start": "constitute a Special",
                "end": "under the provisions of this Chapter.",
            },
            {
                "rule": "85",
                "start": "shall consist of the following members",
                "end": "the convenor of the Special",
            },
            {
                "rule": "85",
                "start": "issue permit as decided by",
                "end": "the Committee.",
            },
        ],
    },
    {
        "id": "kmbr_mh_034",
        "query": "When can buildings be permitted to abut road boundaries under the road-development chapter, what prior approvals/consultations are needed, and to which acquisition cases does this flexibility extend?",
        "evidence": [
            {
                "rule": "85A",
                "start": "permit any building or all buildings upto a particular height to",
                "end": "less than that specified under",
            },
            {
                "rule": "85A",
                "start": "Government",
                "end": "Special Committee.",
            },
            {
                "rule": "85A",
                "start": "Provisions in this rule shall apply to buildings proposed in",
                "end": "Land Acquisition Act also.",
            },
        ],
    },
    {
        "id": "kmbr_mh_035",
        "query": "For accessory sheds and temporary huts, what use/size limits qualify for exemption, what setback/abutting rule applies, and what permit-demolition cycle governs temporary structures?",
        "evidence": [
            {
                "rule": "86",
                "start": "for keeping not more",
                "end": "minimum one metre set back",
            },
            {
                "rule": "86",
                "start": "and may abut",
                "end": "the main building",
            },
            {
                "rule": "87",
                "start": "grant permission",
                "end": "fixed by the Council.",
            },
            {
                "rule": "87",
                "start": "failure of the person to demolish or dismantle the shed or",
                "end": "arrears of property tax due under the Act.",
            },
            {
                "rule": "87",
                "start": "Application for",
                "end": "the land is not owned by the applicant.",
            },
            {
                "rule": "87",
                "start": "issue permit",
                "end": "shall not be retained.",
            },
        ],
    },
    {
        "id": "kmbr_mh_036",
        "query": "For conversion of roofs/shutters in existing buildings, what conversions are allowed without structural/area increase, which general provisions are waived, and what application-fee-validity-completion workflow applies?",
        "evidence": [
            {
                "rule": "91",
                "start": "may permit the conversion",
                "end": "withoutany structural change or area increase.",
            },
            {
                "rule": "92",
                "start": "General provisions regarding",
                "end": "shall not apply to works under this chapter",
            },
            {
                "rule": "93",
                "start": "no application fee for conversion of roof",
                "end": "as in Schedule -",
            },
            {
                "rule": "93",
                "start": "issue permit",
                "end": "Schedule-II.",
            },
            {
                "rule": "94",
                "start": "valid for 6 months from the date of issue",
                "end": "renewal fee shall be fifty % of the permit fee.",
            },
            {
                "rule": "95",
                "start": "report the completion of the work to the Secretary",
                "end": "specifying the date of completion",
            },
        ],
    },
    {
        "id": "kmbr_mh_037",
        "query": "For wall/fence works abutting public interfaces, what pre-commencement prohibition and projection limits apply, and what submission, disposal, validity, renewal, and completion-report rules govern permits?",
        "evidence": [
            {
                "rule": "96",
                "start": "shall not be ,begun unless and until the",
                "end": "execution of the work:",
            },
            {
                "rule": "96",
                "start": "shall not open or project into",
                "end": "property or street.",
            },
            {
                "rule": "97",
                "start": "submitted in",
                "end": "necessary court",
            },
            {
                "rule": "97",
                "start": "issue permit not later",
                "end": "receipt of the application.",
            },
            {
                "rule": "98",
                "start": "valid for one year from the date of issue",
                "end": "renewal fee shall be fifty % of the permit fee.",
            },
            {
                "rule": "99",
                "start": "completion report to the Secretary",
                "end": "date of completion.",
            },
        ],
    },
    {
        "id": "kmbr_mh_038",
        "query": "For additions over legacy buildings, what broad exemptions apply under Rules 100/101, what electrical/open-space/consent constraints still remain, what parking carve-outs exist, and how are opening permissions controlled?",
        "evidence": [
            {
                "rule": "100",
                "start": "buildings existing on the 30th March 2000",
                "end": "Chapter VI and Chapter",
            },
            {
                "rule": "100",
                "start": "clearance from overhead electric lines specified under",
                "end": "shall be produced for issuing permit.",
            },
            {
                "rule": "100",
                "start": "average 60 cms open space from the boundaries",
                "end": "not more than two sides shall be permitted to abut the boundary",
            },
            {
                "rule": "100",
                "start": "no",
                "end": "shall be permitted under",
                "start_occurrence": 5,
            },
            {
                "rule": "100",
                "start": "Off street parking shall be provided as in Table 5 under rule",
                "end": "of floor (s).",
            },
            {
                "rule": "100",
                "start": "door shall be permitted only on the side or portion having 1.00",
                "end": "less than 60 cms open space.",
            },
            {
                "rule": "101",
                "start": "minimum 3 meters",
                "end": "minimum 1.5 meters from the boundary of other road.",
            },
            {
                "rule": "101",
                "start": "sub rules (2) to (10) of rule 100 shall",
                "end": "building.",
            },
        ],
    },
    {
        "id": "kmbr_mh_039",
        "query": "Under Rule 102 flexibility, how are ground/upper-floor additions and separate buildings treated despite existing non-conformity, and what distance/FAR/coverage and roof-shutter conversion constraints still apply?",
        "evidence": [
            {
                "rule": "102",
                "start": "Extension in the ground floor with or without floors above it",
                "end": "shall be taken into account.",
            },
            {
                "rule": "102",
                "start": "Addition/extension of upper floors to any building shall be",
                "end": "be taken into account.",
            },
            {
                "rule": "102",
                "start": "side of cul-de-sac not",
                "end": "rule 25 and rule 32:",
            },
            {
                "rule": "102",
                "start": "Separate and independent building shall be permitted in a plot",
                "end": "has more than three floors.",
            },
            {
                "rule": "102",
                "start": "Conversion of",
                "end": "leaf or thatch:",
            },
            {
                "rule": "102",
                "start": "Conversion of",
                "end": "increased",
                "start_occurrence": 2,
            },
        ],
    },
    {
        "id": "kmbr_mh_040",
        "query": "For new wells, what permit and plan-submission details apply, what sanitary distance constraints govern surrounding pits/tanks, and what are permit validity/renewal and completion-report obligations?",
        "evidence": [
            {
                "rule": "103",
                "start": "No new well shall be dug without",
                "end": "permission of the Secretary.",
            },
            {
                "rule": "103",
                "start": "site plan",
                "end": "within 7.5",
            },
            {
                "rule": "104",
                "start": "set back from any",
                "end": "within 1.20 metres distance from the plot boundaries.",
            },
            {
                "rule": "108",
                "start": "valid for two years and may be renewed for a further period of",
                "end": "permit fee.",
            },
            {
                "rule": "109",
                "start": "report to the Secretary",
                "end": "date of completion.",
            },
        ],
    },
    {
        "id": "kmbr_mh_041",
        "query": "For high-rise apartment proposals, what threshold defines high-rise, what staircase/fire-escape/open-space controls apply, when is stretcher lift mandatory, and what structural documents must accompany application?",
        "evidence": [
            {
                "rule": "110",
                "start": "high rise building means a building having",
                "end": "15 metres of height from ground level.",
            },
            {
                "rule": "112",
                "start": "shall have at least two staircases.",
                "end": "10 cms wide.",
            },
            {
                "rule": "114",
                "start": "Every high rise building shall be provided with a fire escape stairway.",
                "end": "limited to 16 per flight.",
            },
            {
                "rule": "117",
                "start": "if it does not abut on two or more motorable",
                "end": "as specified above need be provided.",
            },
            {
                "rule": "118",
                "start": "having more than 16 dwelling",
                "end": "shall be one capable of carrying a stretcher.",
            },
            {
                "rule": "120",
                "start": "shall be accompanied by one set of structural",
                "end": "prepared and issued by a registered engineer.",
            },
        ],
    },
    {
        "id": "kmbr_mh_042",
        "query": "For hut construction, what permit/setback and rule-waiver framework applies, what 14-day and 30-day decision timelines govern secretary/council action, and what validity and occupancy-certificate deadlines follow completion?",
        "evidence": [
            {
                "rule": "121",
                "start": "No person shall commence the construction of",
                "end": "permit for such",
            },
            {
                "rule": "122",
                "start": "minimum distance between the plot boundary abutting any street",
                "end": "minimum 60 cms distance",
            },
            {
                "rule": "123",
                "start": "FAR, coverage, distance from central line of road",
                "end": "shall not apply to huts.",
            },
            {
                "rule": "125",
                "start": "issue permit within",
                "end": "date of receipt of the application",
            },
            {
                "rule": "126",
                "start": "after",
                "end": "within 30 days from the date of submission of the request in writing.",
            },
            {
                "rule": "128",
                "start": "permit shall be valid for 2 years from the date of issue",
                "end": "payment of fifty % of the permit fee.",
            },
            {
                "rule": "129",
                "start": "issue occupancy certificate not later than 10 days",
                "end": "receipt of the report.",
            },
        ],
    },
    {
        "id": "kmbr_mh_043",
        "query": "For telecom tower siting, what core permit requirement, setback envelope, FAR/height exemptions, defence/railway clearance triggers, and zoning flexibility rules must be jointly satisfied?",
        "evidence": [
            {
                "rule": "130",
                "start": "NO person shall erect or re-erect any non Governmental",
                "end": "from the Secretary.",
            },
            {
                "rule": "131",
                "start": "minimum 3 metres distance from",
                "end": "distance:",
            },
            {
                "rule": "131",
                "start": "Distance from",
                "end": "minimum 1.20",
            },
            {
                "rule": "131",
                "start": "consent of the owner of the plot on the",
                "end": "application",
            },
            {
                "rule": "132",
                "start": "shall",
                "end": "produced before issuing permit.",
                "start_occurrence": 1,
            },
            {
                "rule": "140",
                "start": "within 200 metres",
                "end": "before",
            },
            {
                "rule": "140A",
                "start": "No site approval shall be",
                "end": "irrespective of its occupancy.",
            },
        ],
    },
    {
        "id": "kmbr_mh_044",
        "query": "For telecom permit processing and post-completion use, what document package and engineer credentials are required, what fees/validity-extension charges apply, and what certificates must be filed before use certificate issue?",
        "evidence": [
            {
                "rule": "141",
                "start": "Application",
                "end": "document to prove ownership.",
            },
            {
                "rule": "141",
                "start": "structural",
                "end": "Quasi- Government Organisation.",
                "start_occurrence": 3,
            },
            {
                "rule": "141",
                "start": "for towers",
                "end": "rupees ten thousand;",
            },
            {
                "rule": "141",
                "start": "for pole structures",
                "end": "five",
            },
            {
                "rule": "141",
                "start": "permit",
                "end": "date of issue of permit.",
                "start_occurrence": 8,
            },
            {
                "rule": "141",
                "start": "extended for a further period of one year",
                "end": "valid period of the permit.",
            },
            {
                "rule": "141",
                "start": "fee for extension shall be equal to fifty % of the fee for",
                "end": "time of extension.",
            },
            {
                "rule": "142",
                "start": "After completion of",
                "end": "for use of the service].",
            },
        ],
    },
    {
        "id": "kmbr_mh_045",
        "query": "In regularization proceedings, what eligibility and deemed-permitted carveout rules apply, what filing package and compounding computation are required, and what demolition/prosecution consequences follow non-compliance or refusal?",
        "evidence": [
            {
                "rule": "143",
                "start": "power",
                "end": "in deviation",
            },
            {
                "rule": "143",
                "start": "shall not be in violation",
                "end": "these rules.",
            },
            {
                "rule": "143",
                "start": "after the statutory period specified in rules 15,",
                "end": "not one requiring regularization.",
            },
            {
                "rule": "144",
                "start": "application for regularization shall be accompanied by documentary",
                "end": "application for new permit.",
            },
            {
                "rule": "146",
                "start": "compounding fee shall be double the amount of the permit fee",
                "end": "calculation",
            },
            {
                "rule": "146",
                "start": "application for regularisation shall be refused only on",
                "end": "may be refused.",
            },
            {
                "rule": "147",
                "start": "fails to remit the compounding fee within the time specified",
                "end": "arrear of property",
            },
            {
                "rule": "147",
                "start": "shall not be demolished or well filled up or prosecution",
                "end": "has not been expired.",
            },
        ],
    },
    {
        "id": "kmbr_mh_046",
        "query": "For professional signatories under KMBR, who must be registered, who is the registering authority, who is ineligible by employment status, what validity/renewal regime applies, and how does multi-category registration work?",
        "evidence": [
            {
                "rule": "148",
                "start": "wherever it is",
                "end": "under the provisions in this chapter.",
            },
            {
                "rule": "149",
                "start": "Registering Authority for",
                "end": "the State.",
            },
            {
                "rule": "150",
                "start": "A person employed in the service of Government or Quasi-Government",
                "end": "for this purpose.",
            },
            {
                "rule": "150",
                "start": "registration once made shall be valid for three years",
                "end": "valid period of registration:",
            },
            {
                "rule": "151",
                "start": "No person shall be eligible for registration in the category",
                "end": "of Appendix-L.",
            },
            {
                "rule": "152",
                "start": "eligible for registration in more than one category",
                "end": "registration in each such category.",
            },
        ],
    },
]


def chapter_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"chapter(\d+)", path.stem)
    if not m:
        return (9999, path.name)
    return (int(m.group(1)), path.name)


def write_combined_source() -> list[str]:
    chapters = sorted(CHAPTER_DIR.glob("chapter*.md"), key=chapter_sort_key)
    if not chapters:
        raise FileNotFoundError(f"No chapter markdown files found under {CHAPTER_DIR}")

    parts: list[str] = []
    for chapter in chapters:
        parts.append(f"<!-- BEGIN {chapter.as_posix()} -->")
        parts.append(chapter.read_text(encoding="utf-8").rstrip())
        parts.append(f"<!-- END {chapter.as_posix()} -->")
        parts.append("")

    text = "\n".join(parts).rstrip() + "\n"
    COMBINED_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_SOURCE.write_text(text, encoding="utf-8")
    return text.splitlines()


def build_rule_blocks(lines: list[str]) -> dict[str, tuple[int, int]]:
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        m = RULE_HEADER_RE.match(line)
        if m:
            starts.append((m.group(1), idx))

    if not starts:
        raise ValueError("No rule headers found in combined source")

    blocks: dict[str, tuple[int, int]] = {}
    for i, (rule, start_line) in enumerate(starts):
        end_line = starts[i + 1][1] - 1 if i + 1 < len(starts) else len(lines)
        blocks[rule] = (start_line, end_line)
    return blocks


def find_line(lines: list[str], start: int, end: int, needle: str, occurrence: int = 1) -> int:
    count = 0
    needle_l = needle.strip().lower()
    for lineno in range(start, end + 1):
        if needle_l in lines[lineno - 1].lower():
            count += 1
            if count == occurrence:
                return lineno
    raise ValueError(f"Could not find '{needle}' occurrence {occurrence} in lines {start}-{end}")


def extract_snippet(lines: list[str], start: int, end: int) -> str:
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"Invalid line range {start}-{end}")
    return "\n".join(lines[start - 1 : end])


def resolve_evidence(
    lines: list[str],
    rule_blocks: dict[str, tuple[int, int]],
    spec: dict[str, Any],
) -> tuple[int, int]:
    rule = str(spec["rule"])
    if rule not in rule_blocks:
        raise KeyError(f"Rule {rule} not found in combined source")

    block_start, block_end = rule_blocks[rule]
    start_occ = int(spec.get("start_occurrence", 1))
    end_occ = int(spec.get("end_occurrence", 1))

    start_line = find_line(lines, block_start, block_end, spec["start"], start_occ)

    if "end" in spec and spec["end"]:
        end_line = find_line(lines, start_line, block_end, spec["end"], end_occ)
    else:
        end_line = start_line

    before = int(spec.get("before", 0))
    after = int(spec.get("after", 0))
    start_line = max(block_start, start_line + before)
    end_line = min(block_end, end_line + after)

    if end_line < start_line:
        raise ValueError(f"Resolved invalid range {start_line}-{end_line} for rule {rule}")

    return start_line, end_line


def build_dataset() -> list[dict[str, Any]]:
    lines = write_combined_source()
    rule_blocks = build_rule_blocks(lines)

    dataset: list[dict[str, Any]] = []
    for row in ENTRIES:
        chunks: list[dict[str, str]] = []
        for spec in row["evidence"]:
            start, end = resolve_evidence(lines, rule_blocks, spec)
            chunks.append(
                {
                    "citation": f"{COMBINED_SOURCE.as_posix()}:{start}-{end}",
                    "snippet": extract_snippet(lines, start, end),
                }
            )

        dataset.append(
            {
                "id": row["id"],
                "query": row["query"],
                "ground_truth_chunks": chunks,
                "source_document": COMBINED_SOURCE.as_posix(),
                "query_type": "multi_hop_complex",
            }
        )

    return dataset


def main() -> None:
    dataset = build_dataset()

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    OUT_JSON.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(dataset)} records")
    print(f"- {OUT_JSONL}")
    print(f"- {OUT_JSON}")
    print(f"- {COMBINED_SOURCE}")


if __name__ == "__main__":
    main()
