#!/usr/bin/env python3
"""
Clean the 1950 Survey of Consumer Finances (ICPSR 3612) card-image file.

    python3 clean_scf1950.py

Input : 03612-0001-Data-card_image.txt  (38,632 lines = 3,512 spending units x 11 punch cards)
Output: clean/scf1950_clean.csv       one row per spending unit, codes kept, dollars decoded
        clean/scf1950_labeled.csv     same, with categorical codes replaced by their labels
        clean/scf1950_variables.csv   data dictionary (card, columns, type, labels, notes)
        clean/cleaning_report.txt     every check and every repair that was made

Column positions were taken from the original SRC codebook (codebook_small.txt / the PDF),
NOT from the AI-generated tables in "Codebook SCF 1950 Final Version.xlsx", many of which
are shifted or wrong. Each position was then checked against the data (see the report).

Punch-card conventions handled here
  '-'  is the X punch (zone 11)  -> written as 'X'   (usually "not ascertained")
  '&'  is the Y punch (zone 12)  -> written as 'Y'   (usually "not ascertained whether" / inapplicable)
  Dollar fields are punched in $1, $10 or $100 units; all are converted to dollars.
  A dollar field containing an X or Y punch is set to missing and the reason is
  stored in <name>_status:
      amount_na   leading X : has the item, amount not ascertained
      over_field  leading Y : amount too big for the field (patched from the codebook's
                              "Cases in which Amount Exceeded the Field" list where given)
      na_whether  trailing X: not ascertained whether has the item at all
      negative / loss / farm_schedule : trailing Y, meaning depends on the field
      under_50    trailing Y in a $100-unit field: "less than $50" -> value set to 25
      inapplicable  house_value, monthly_rent, house_plan_price, car_price, car_plan_price only:
                    a punched 0 means "does not own / rent / plan to buy", so it is set to missing
  All other dollar fields keep 0 as a true zero (no income, no assets, no debt, no spending).

To analyse the result, load it with the right column types (codes such as '01' must stay text):

    from clean_scf1950 import load
    df = load()            # or load(labeled=True) for label text instead of codes
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "03612-0001-Data-card_image.txt")
OUT = os.path.join(HERE, "clean")
CARDS = "123456789YZ"


# --------------------------------------------------------------------------- labels
def L(spec):
    """'1=yes;5=no' -> {'1': 'yes', '5': 'no'}"""
    out = {}
    for part in spec.split(";"):
        part = part.strip()
        if part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def bracket(spec, extra="0=none;X=amount not ascertained;Y=not ascertained whether"):
    """'1-99|100-499' -> {'1': '$1-99', '2': '$100-499', ...} plus the usual 0/X/Y codes"""
    out = {str(i + 1): "$" + b for i, b in enumerate(spec.split("|")) if b}
    out.update(L(extra))
    return out


YEARS = {"35": "1935 or earlier", "36": "1936", "37": "1937", "38": "1938", "39": "1939",
         "3X": "1930s / very old", "40": "1940", "41": "1941", "42": "1942", "43": "1943",
         "44": "1944", "55": "1945", "56": "1946", "57": "1947", "58": "1948", "59": "1949",
         "60": "1950", "99": "don't know", "00": "inapplicable (no car / no purchase)",
         "X0": "not ascertained", "Y0": "not ascertained whether owns"}
YEAR_NUM = {"35": 1935, "36": 1936, "37": 1937, "38": 1938, "39": 1939, "40": 1940, "41": 1941,
            "42": 1942, "43": 1943, "44": 1944, "55": 1945, "56": 1946, "57": 1947, "58": 1948,
            "59": 1949, "60": 1950}
OCC = L("11=professional and semi-professional;21=self-employed businessman;"
        "22=manager or official (not self-employed, not farm);23=farm manager;"
        "24=self-employed artisan or craftsman;2X=businessman/manager/artisan, NA which;"
        "31=clerical, sales clerk, attendant;32=buyer, agent, broker;41=skilled worker, foreman;"
        "42=semi-skilled worker, operative;4X=skilled or semi-skilled, NA which;"
        "51=unskilled laborer (not farm);52=farm laborer;53=domestic service (private family);"
        "54=service worker (not domestic);5X=unskilled or service, NA which;"
        "61=protective service;71=unemployed - professional;72=unemployed - proprietor/manager;"
        "73=unemployed - clerical/sales;74=unemployed - skilled/semi-skilled;"
        "75=unemployed - unskilled/service;76=unemployed - protective service;"
        "78=unemployed - farm operator;7X=unemployed - occupation not given;88=farm operator;"
        "98=retired but works a little;99=retired;00=student;01=student with part-time work;"
        "YY=housewife;Y8=housewife but works a little (code as punched, codebook illegible);"
        "XX=not ascertained")
REASON_TIMES = L("11=supply situation;12=demand situation (not purchasing power);13=purchasing power;"
                 "14=production costs incl. labor costs;15=wages, general;16=employment;"
                 "17=business or farm profits;18=crops, weather, livestock;1X=business conditions, general;"
                 "21=high prices;22=low prices;2X=prices, general;31=politics or government;"
                 "32=labor-management relations;33=war;34=working conditions;41=personal income;"
                 "42=personal employment situation;43=other personal factors;77=other;"
                 "00=no reason / no second reason;XX=not ascertained")
INC8 = L("1=$0-999;2=$1,000-1,999;3=$2,000-2,999;4=$3,000-3,999;5=$4,000-4,999;6=$5,000-7,499;"
         "7=$7,500-9,999;8=$10,000 and over;9=negative;0=no income;X=not ascertained")
SRC8 = bracket("1-99|100-499|500-999|1,000-1,999|2,000-2,999|3,000-4,999|5,000-9,999|10,000 and over",
               "9=negative amount;0=none;X=amount not ascertained;Y=not ascertained whether")
SRC9 = bracket("1-99|100-499|500-999|1,000-1,999|2,000-2,999|3,000-3,999|4,000-4,999|5,000-9,999|10,000 and over")
LIQ8 = bracket("1-199|200-499|500-999|1,000-1,999|2,000-4,999|5,000-9,999|10,000-24,999|25,000 and over")
LIQ6 = bracket("1-199|200-499|500-999|1,000-1,999|2,000-4,999|5,000 and over", "9=not ascertained whether;0=none;X=amount not ascertained;Y=not ascertained whether")
LIQ5 = bracket("1-199|200-499|500-999|1,000-1,999|2,000 and over")
CHG9 = L("1=increase $1,000 or more;2=increase $500-999;3=increase $200-499;4=increase $1-199;"
         "5=has account, no change;6=decrease $1-199;7=decrease $200-499;8=decrease $500-999;"
         "9=decrease $1,000 or more;0=none now and none a year ago;X=change not ascertained;"
         "Y=not ascertained whether has/had")
DEBT8 = bracket("1-949|950-1,949|1,950-2,949|2,950-3,949|3,950-4,949|4,950-7,449|7,450-9,949|9,950 and over")
HOUSE9 = bracket("under 2,500|2,500-4,999|5,000-7,499|7,500-9,999|10,000-12,499|12,500-14,999|"
                 "15,000-19,999|20,000-29,999|30,000 and over",
                 "0=inapplicable;X=amount not ascertained;Y=not ascertained whether")
HOUSE9["1"] = "under $2,500"
PAY7 = bracket("1-99|100-199|200-299|300-499|500-999|1,000-1,499|1,500 and over",
               "8=nothing paid (on principal);9=not ascertained whether paid;0=inapplicable;"
               "X=paid, amount not ascertained;Y=not ascertained")
B15 = bracket("1-99|100-299|300-499|500-999|1,000-1,499|1,500-1,999|2,000-4,999|5,000 and over")
B_SMALL6 = bracket("1-199|200-499|500-999|1,000-1,999|2,000-4,999|5,000 and over", "0=zero;X=amount not ascertained;Y=not ascertained whether")
RE8 = bracket("1-499|500-999|1,000-1,999|2,000-2,999|3,000-4,999|5,000-7,499|7,500-9,999|10,000 or more")
SAVPCT = L("1=plus 50% or more;2=plus 30-49%;3=plus 20-29%;4=plus 10-19%;5=plus 1-9%;6=minus 1-9%;"
           "7=minus 10-24%;8=minus 25% or more;0=zero saving / no or negative income;X=not ascertained")
SAVCLS = L("1=plus $2,000 or more;2=plus $1,000-1,999;3=plus $500-999;4=plus $200-499;5=plus $100-199;"
           "6=plus $1-99;7=minus $1-99;8=minus $100-499;9=minus $500 or more;0=zero;X=not ascertained")
SAVSIGN = L("1=positive (not due to accrual only);2=positive, due to accrual on A-F bonds only;"
            "3=negative;0=zero;X=not ascertained")
RENTPCT = L("1=under 5%;2=5-9%;3=10-14%;4=15-19%;5=20-24%;6=25-29%;7=30-34%;8=35-39%;9=40% or more;"
            "0=does not rent;X=percent not ascertained (or zero/negative income);Y=not ascertained whether rents")
PCTVAL = L("1=1% and under;2=2%;3=3%;4=4%;5=5-9%;6=10-14%;7=15-19%;8=20-24%;9=25% or more;"
           "0=none / does not own;X=percent not ascertained;Y=not ascertained whether")
DUR8 = L("1=$1-34;2=$35-74 ($1-74 for refrigerator, TV, furniture);3=$75-174;4=$175-274;5=$275-374;"
         "6=$375-474;7=$475-749;8=$750 or over;9=$1,000 or over (expected TV only);0=none;"
         "X=price not ascertained;Y=not ascertained whether")
CASHPROP = L("1=under 50%;2=50-99%;3=100%, all cash;0=no purchase;X=not ascertained;Y=not ascertained whether bought")
TRADEIN = L("1=trade-in under $50;2=trade-in $50-99;3=trade-in $100 or more;4=trade-in, value NA;"
            "9=purchased, no trade-in;0=not purchased;X=not ascertained;Y=not ascertained whether bought")
DURTYPE = L("1=furniture;2=refrigerator, ice box, deep freeze;3=radio, record player;4=television set;"
            "5=washing machine;6=stove;7=other electrical appliances;8=other items;9=more than two categories;"
            "0=none / no second item;X=bought, nature not ascertained;Y=not ascertained whether")
HALF = L("1=first half of 1950;2=second half of 1950;3=uncertain which half;4=has already bought in 1950;"
         "9=don't know;0=won't buy;X=not ascertained which half;Y=not ascertained whether will buy")
FINPLAN = L("1=installment (plus down payment);2=borrow some other way;3=cash only;9=don't know;"
            "0=will not buy;X=not ascertained;Y=not ascertained whether will buy")
CARPLAN = L("1=will buy new;2=will buy used;3=will buy, uncertain new/used;4=probably will buy new;"
            "5=probably will buy used;6=probably will buy, uncertain new/used;7=undecided - new;"
            "8=undecided - used;9=undecided - uncertain new/used;0=will not buy;X=not ascertained")
CAROWN = L("1=owns one car;2=owns two cars;3=owns three or more;4=no, but owned in 1949;"
           "0=does not own, did not own in 1949;X=not ascertained")
CARNEW = L("1=owns one, bought new;2=owns one, bought used;3=owns 2+, bought new;4=owns 2+, bought used;"
           "5=owns 2+, mixed or NA;6=owned in 1949, bought new;7=owned in 1949, bought used;"
           "8=owned in 1949, NA new/used;0=no car;X=owns, NA new/used;Y=not ascertained whether owns")
PLACE10 = L("1=metropolitan central city;2=metropolitan suburb;3=rural metropolitan county;"
            "4=city 50,000+ (not metropolitan);5=city 2,500-50,000, South;6=city 2,500-50,000, not South;"
            "7=town under 2,500, South;8=town under 2,500, not South;9=open country, South;"
            "0=open country, not South;X=not ascertained")
AGE = L("1=18-24;2=25-34;3=35-44;4=45-54;5=55-64;6=65 or over;X=not ascertained")
EDUC = L("1=grammar school (1-8);2=high school (9-12) or other non-college;3=college;0=none;X=not ascertained")
VET = L("1=head is the only WWII veteran;2=one veteran, not the head;3=two or more veterans incl. head;"
        "4=two or more veterans, not the head;0=no veteran;X=not ascertained")
CHILD = L("1=one;2=two;3=three;4=four;5=five;6=six;7=seven or more;0=none;X=not ascertained")
PEOPLE = L("1=one;2=two;3=three;4=four;5=five;6=six;7=seven;8=eight or more;X=not ascertained")
N4 = L("1=one;2=two;3=three;4=four or more;0=none;X=not ascertained")
FILTER_FAM = L("1=single-SU family or primary, complete family data;2=related secondary with family data (substitute);"
               "3=related secondary - family data coded under primary's interview number;"
               "4=single-SU family or primary, MISSING family data;5=SU composition NA, family data included;"
               "6=SU composition NA, missing family data")

SAMPLE_POINTS = dict(zip(
    ["%02d" % i for i in range(1, 74)],
    ["Baltimore", "Boston", "Chicago", "Chicago", "Cleveland", "Detroit", "Detroit", "Los Angeles",
     "Los Angeles", "New York - Bronx / Newark", "New York - Brooklyn",
     "New York - Manhattan / Montclair / Mt. Vernon / Neptune",
     "New York - Queens / Bayonne / Long Beach / Mineola",
     "New York area - Nutley / Paterson / Totowa / Yonkers / Monmouth / Nassau", "Philadelphia",
     "Pittsburgh", "San Francisco", "St. Louis", "Washington, D.C.", "Adair, Mo.", "Augusta, Va.",
     "Beadle, S. Dak.", "Berks, Penn.", "Berkshire, Mass.", "Black Hawk, Iowa", "Boone, Ill.",
     "Boyd, Ky.", "Buchanan, Va.", "Bucks, Penn.", "Carter, Tenn.", "Clark, Ark.", "Covington, Ky.",
     "Crawford, Iowa", "Denver, Colo.", "Erie, N.Y.", "Fairfield, Conn.", "Forsyth, N.C.",
     "Genesee, Mich.", "Geneva, Ala.", "Grays Harbor, Wash.", "Grayson, Tex.", "Hamilton, Ohio",
     "Hamilton, Tenn.", "Harris, Tex.", "Hennepin, Minn.", "Hyde, N.C.", "Idaho, Idaho",
     "Jeff Davis, Ga.", "Jefferson, Ark.", "Lackawanna, Penn.", "Lancaster, S.C.", "Lubbock, Tex.",
     "Mahoning, Ohio", "Marion, Ill.", "Marion, Ind.", "Mitchell, Kan.", "Multnomah, Ore.",
     "Newton, Miss.", "Noble, Okla.", "Oklahoma, Okla.", "Onondaga, N.Y.", "Page, Iowa", "Pitt, N.C.",
     "Posey, Ind.", "Seneca, Ohio", "St. Clair, Mich.", "St. Lucie, Fla.", "Susquehanna, Penn.",
     "Traverse, Minn.", "Union, Ohio", "Weld, Colo.", "Westmoreland, Penn.", "Windsor, Vt."]))

LABELS = {
    "YN": L("1=yes;5=no;9=uncertain, don't know;X=not ascertained;Y=not ascertained whether"),
    "ASSIGNED": L("1=amount was assigned (imputed by SRC);5=amount not assigned;9=not ascertained"),
    "RATIO": L("1=nonfarm group 1 (basic sampling rate);2=nonfarm group 2 (2x basic rate);"
               "6=nonfarm group 6 (6x basic rate);7=farm schedule, segment F;"
               "8=farm schedule, other open-country segments;9=not ascertained"),
    "PLACE10": PLACE10,
    "SUCOMP": L("01=primary SU, no secondaries;12=primary with 1 unrelated secondary;13=primary with 2 unrelated;"
                "14=primary with 3+ unrelated;21=secondary, unrelated - roomer;22=secondary, unrelated - servant;"
                "23=secondary, unrelated - other;31=primary with 1 related secondary;32=primary with 2 related;"
                "33=primary with 3+ related;34=primary with 1 related and 1 unrelated;"
                "35=primary with related and unrelated (more than 1);41=secondary, adult child of head;"
                "42=secondary, parent(-in-law) of head;43=secondary, otherwise related;XX=not ascertained"),
    "DATE": L("1=January 1-15, 1950;2=January 16-31;3=February 1-14;4=February 15-28;5=March 1 or later;X=not ascertained"),
    "RHEAD": L("1=respondent is head of SU;2=respondent is wife or other member;3=head and wife both responding;"
               "4=respondent outside the SU;X=not ascertained"),
    "N4": N4,
    "WHYNOTSU": L("1=no income or under $10 a week;2=does not keep finances separate;"
                  "3=keeps finances separate but contributes half or more;0=no other adult / no second reason;X=not ascertained"),
    "SUCHANGE": L("1=increase or decrease of adults;2=no change in adults;X=not ascertained"),
    "CHILD": CHILD, "PEOPLE": PEOPLE,
    "BORN": L("1=yes;5=no;9=don't know;0=no children under 18;X=not ascertained"),
    "DEPEND": L("1=one;2=two;3=three or more;0=none;X=not ascertained"),
    "MARRIED": L("1=less than 1 year;2=1 year;3=2 years;4=3 years;5=4 years;6=5-9 years;7=10-20 years;"
                 "8=over 20 years;9=don't know;X=not ascertained;0=not married (single, divorced, widowed, separated)"),
    "AGE": AGE, "EDUC": EDUC, "VET": VET,
    "SEX": L("1=male;2=female;X=not ascertained"),
    "RACE": L("1=white;2=Negro;3=other;X=not ascertained (includes Mexican)"),
    "CALLS": L("1=one;2=two;3=three;4=four;5=five;6=six;7=more than six;X=not ascertained"),
    "OCC": OCC,
    "SELFEMP": L("1=self-employed;2=works for someone else;3=both;8=farm schedule;0=not working;X=not ascertained"),
    "MONTHS": L("1=11-12 months (whole year);2=9-10 months;3=7-8 months;4=5-6 months;5=3-4 months;6=1-2 months;"
                "7=none, unemployed all year;8=farm schedule;9=don't know;"
                "0=inapplicable (self-employed, student, housewife);X=not ascertained"),
    "OWNBUS": L("1=owns a business now;3=owned one during 1949;5=does not own;9=don't know;X=not ascertained"),
    "BUSTYPE": L("11=grocery or other food store;12=restaurant, bar;13=clothing store;14=other retail store;"
                 "21=filling station;22=beauty or barber shop;23=shoe repair;24=laundry, cleaners;"
                 "25=auto dealer, repair shop, garage;31=real estate, insurance, finance;32=contractor;"
                 "33=trucking or taxi;41=food manufacturing;42=printing or publishing;49=other manufacturing;"
                 "51=wholesaler;61=general contractor;62=self-employed artisan;77=other;"
                 "XX=type not ascertained;YY=not ascertained whether owns;00=no business"),
    "CORP": L("1=incorporated;2=unincorporated;3=both;9=don't know;0=no business;X=owns, form NA;Y=not ascertained whether owns"),
    "PLACE6": L("1=metropolitan area;2=city 50,000 and over (non-metropolitan);3=city or town 2,500-50,000;"
                "4=town under 2,500;5=open country;6=metropolitan rural"),
    "INC8": INC8,
    "OCC10": L("1=professional and semi-professional;2=self-employed businessmen and artisans;"
               "3=managers and officials;4=white collar (clerical and sales);5=skilled or semi-skilled;"
               "6=unskilled;7=farm laborers;8=farm operators;9=unemployed;"
               "0=students, housewives, protective service;X=not ascertained;Y=retired"),
    "FAMSTAT": L("1=young (under 45), not married, no children;2=young, married, no children;"
                 "3=old (45+), married, no children;4=old, unmarried, no children;5=married, children, head under 35;"
                 "6=unmarried, children;7=unmarried, no children, age NA;8=married, no children, age NA;"
                 "9=married, children, head 35-44;0=married, children, head 45+;X=not ascertained"),
    "SPLIT": L("01=Baltimore city;02=Boston city;03=Chicago city;04=Cleveland city;05=Detroit city;"
               "06=Los Angeles city;07=New York city;08=Philadelphia city;09=Pittsburgh city;"
               "10=San Francisco city;11=St. Louis city;12=Washington city;13=New York suburbs;"
               "14=Eastern metropolitan suburbs;15=Midwest metropolitan suburbs;16=Pacific metropolitan suburbs;"
               "99=not ascertained;" + ";".join(f"{k}={v}" for k, v in SAMPLE_POINTS.items() if k >= "20")),
    # ---- card 2
    "BETTER3": L("1=better off;3=no change;5=worse off;9=undecided, don't know;X=not ascertained"),
    "MORE3": L("1=making more now;3=about the same;5=making less now;8=farm schedule;9=don't know;X=not ascertained"),
    "WILL3": L("1=will make more;3=about the same;5=will make less;8=farm schedule;9=don't know, it depends;X=not ascertained"),
    "REASON_FIN": L("11=higher income: raise or promotion;12=higher income: better-paying job;"
                    "13=higher income: steadier work, overtime;14=higher income: more members working;"
                    "15=higher income: business, farm, professional profits;16=higher income: outside contributions;"
                    "1X=higher income, general;21=lower prices;22=lower expenses;23=greater savings, house bought, bills paid;"
                    "24=taxes reduced;25=greater security;26=greater opportunities;27=liquidated assets;33=other (better);"
                    "41=lower income: cut in wages;42=lower income: lower-paying job;"
                    "43=lower income: unsteady work, unemployment, strike;44=lower income: fewer members working;"
                    "45=lower income: business, farm, professional profits;46=lower income: outside contributions;"
                    "47=lower income: head retired, sick, deceased;4X=lower income, general;51=higher prices;"
                    "52=increased expenses;53=savings used up, increased debt;54=taxes burdensome;55=less security;"
                    "56=no opportunities;57=government action;66=other (worse);00=none / no second reason;XX=not ascertained"),
    "REASON_EXP": L("11=change to higher-paying job;12=raise or promotion;13=business more profitable;"
                    "14=more members working;15=steadier job;16=expect to get a job;17=additional job, overtime;"
                    "18=increased outside contributions;19=less illness;22=other (increase);"
                    "31=change to lower-paying job;32=cut in wages, less overtime;33=business less profitable;"
                    "34=fewer members working;35=less steady work;36=head unemployed;37=head retired;"
                    "38=decreased outside contributions;39=illness;44=other (decrease);51=depends on change of job;"
                    "52=depends on wages;53=depends on business or farm profits;54=depends on members working;"
                    "55=depends on steadiness of employment;56=depends on getting a job;57=depends on retirement;"
                    "58=depends on general business conditions;59=depends on government policy;"
                    "61=depends on outside contributions;66=depends on other;88=farm schedule;"
                    "00=none / no change expected;XX=not ascertained"),
    "TIMES5": L("1=very good;2=fairly good;3=middle position;4=fairly bad;5=very bad;9=undecided, don't know;X=not ascertained"),
    "TIMESCHG5": L("1=much better;2=somewhat better;3=middle position;4=somewhat worse;5=much worse;9=don't know;X=not ascertained"),
    "REASON_TIMES": REASON_TIMES,
    "PRICES": L("1=prices will go up;3=stay about the same;5=prices will go down;"
                "6=stay the same or go up (code inferred, codebook illegible);7=stay the same or go down (code inferred, codebook illegible);9=undecided, don't know;X=not ascertained"),
    "REASON_PRICE": L("11=goods still scarce;12=wages will rise or are high;13=people have plenty of money;"
                      "14=government action;15=exports, Marshall Plan, rearmament;16=crops, livestock, weather;"
                      "17=price rigidities;1X=prices rising, general;31=depends on volume of production;"
                      "32=depends on production costs;33=depends on purchasing power;34=depends on consumer resistance;"
                      "35=depends on government action;36=depends on labor-management relations;"
                      "37=depends on weather, crops;38=depends on business action;39=depends on foreign affairs;"
                      "51=production will be ample;52=production costs will fall;53=consumers won't pay high prices;"
                      "54=purchasing power too low or will fall;55=prices fall after every inflation / war;"
                      "56=government will prevent inflation;57=small exports;58=crops, livestock, weather;"
                      "5X=prices falling, general;99=don't know;00=none / no second factor;XX=not ascertained"),
    "PRICEAMT": L("1=a lot higher (11% or more);2=a little higher (1-10%);3=same;4=a little lower (1-10%);"
                  "5=a lot lower (11% or more);9=undecided, don't know;X=not ascertained"),
    # ---- card 3
    "TENURE": L("1=owns home;2=pays rent, primary SU;3=pays rent, unrelated secondary;4=neither owns nor rents;"
                "5=joint owner, cooperative;6=shares rent;7=owns apartment building R lives in;"
                "9=sold house, still lives there;X=not ascertained"),
    "NEITHER": L("1=servant;2=farm laborer;3=other housing as part of compensation;4=other;0=owns or rents;"
                 "X=not ascertained;Y=not ascertained whether rents or owns"),
    "FURN": L("1=furnished;2=partially furnished;3=unfurnished;9=don't know;0=does not rent;X=not ascertained;Y=NA whether rents"),
    "RENTINCL": L("1=utilities;2=no utilities, nothing;3=use of durable goods;4=use of other rooms;5=food, board;"
                  "6=service;7=storage, garage, business rooms;8=everything;9=house and lot;"
                  "0=does not rent / no second item;X=not ascertained;Y=NA whether rents"),
    "RENTOUT": L("1=rents out apartment(s);2=rents out room(s);3=rents out other accommodation;"
                 "7=does not rent out any part (code inferred, codebook illegible);0=does not own;"
                 "X=not ascertained;Y=NA whether rents or owns"),
    "BOUGHT": L("1=before 1920;2=1920-1929;3=1930-1939;4=1940-1944;5=1945;6=1946-1947;7=1948;"
                "8=1949;9=1950;0=does not own;"
                "X=not ascertained / inherited;Y=NA whether rents or owns"),
    "NEWBUILT": L("1=lived in before;2=newly built;0=does not own;X=not ascertained;Y=NA whether rents or owns"),
    "BUILT": L("1=before 1920;2=1920-1929;3=1930-1939;4=1940-1944;5=1945-1947;7=1948;8=1949;9=1950;"
               "0=does not own / inapplicable;X=not ascertained;Y=NA whether rents or owns"),
    "CERTAIN": L("1=quite sure;2=fairly sure;3=not very sure;4=not at all sure, rough guess;5=could not estimate;"
                 "6=refused estimate;9=bought in 1949 or 1950;0=does not own;X=not ascertained;Y=NA whether owns"),
    "KNOWPRIN": L("1=yes;2=no;0=inapplicable;X=not ascertained whether paid;Y=not ascertained"),
    "PRE49": bracket("1-99|100-299|300-499|500-999|1,000-4,999|5,000 and over", "8=bought 1949-50, nothing paid before 1949 (code inferred);0=did not purchase in 1949-50 / does not own;"
                     "X=not ascertained / no housing data on this record;Y=not ascertained"),
    "HOUSE9": HOUSE9,
    "PROPPAID": L("1=paid 50% or more before 1949;2=paid less than 50% before 1949;4=paid nothing before 1949;0=did not buy in 1949-50;"
                  "X=paid, amount NA;Y=not ascertained"),
    "HDEBT49": L("1=mortgage or land contract only;2=other borrowing only, under 10% of price;"
                 "3=other borrowing only, 10% or more;4=mortgage plus other borrowing under 10%;"
                 "5=mortgage plus other borrowing 10% or more;6=no debt incurred;7=mortgage, percent NA;"
                 "0=did not buy in 1949-50 / does not own;X=not ascertained;Y=not ascertained / no housing data on this record"),
    "FUNDS": L("1=no source other than borrowing;2=sold house or other property;3=drew down liquid assets;"
               "4=inheritance;5=gifts from friends or relatives;6=income or saving;7=sale of securities;"
               "8=more than two sources;9=other;0=did not buy / no second source;X=not ascertained;Y=NA whether bought"),
    "REPAIRS": L("1=redecorating interior;2=structural changes inside;3=heating plant;4=fixtures and equipment;"
                 "5=exterior improved;6=additions built;7=landscaping;8=new utilities;9=other;"
                 "0=none / no second item;X=type not ascertained;Y=not ascertained whether"),
    "UNCLASS": bracket("1-49|50-149|150-249|250-499|500-749|750-999|1,000 and over"),
    "REPDEBT": bracket("1-49|50-99|100-199|200-299|300-499|500-999|1,000 and over"),
    "DEBTPCT": L("1=under 20%;2=20-29%;3=30-39%;4=40-49%;5=50-59%;6=60-69%;7=70-79%;8=80% or more;"
                 "9=owns, no mortgage;0=does not own;X=percent NA;Y=not ascertained whether"),
    "PRINPCT": L("1=1-19%;2=20-29%;3=30-39%;4=40-49%;5=50-59%;6=60-69%;7=70-79%;8=80% or more;"
                 "9=has debt, no payment on principal;0=inapplicable;Y=not ascertained"),
    "PCTVAL": PCTVAL, "RENTPCT": RENTPCT,
    "RENTCLS": bracket("1-19|20-29|30-39|40-49|50-74|75-99|100-149|150-199|200 and more",
                       "0=does not rent;X=rent not ascertained;Y=NA whether rents or owns"),
    "HDEBTCLS": bracket("1-499|500-999|1,000-2,999|3,000-4,999|5,000-7,499|7,500-9,999|10,000 or over",
                        "8=owns, no mortgage;9=owns, NA whether has debt;0=does not own;X=amount NA;Y=not ascertained"),
    "PAY7": PAY7,
    "HPURCH": L("1=bought newly built house in 1949;2=bought used house in 1949;3=bought in 1949, NA new/used;"
                "4=bought newly built in 1950;5=bought used in 1950;6=bought in 1950, NA new/used;"
                "8=owns, bought before 1949;X=owns, year NA;0=does not own;Y=housing status not ascertained"),
    "UPKEEP": L("1=additions only;2=additions and repairs;3=additions and unclassifiable;4=unclassifiable only;"
                "5=repairs only;6=repairs and unclassifiable;7=additions, repairs and unclassifiable;0=none;Y=not ascertained"),
    "FILTER_H": L("1=nonfarm single-SU family or primary, complete housing data;2=nonfarm related secondary with housing data;"
                  "3=nonfarm related secondary - housing coded under primary;4=missing housing data;"
                  "5=farm schedule, no housing data;6=farm schedule with housing data;"
                  "7=SU composition NA, housing data included;8=SU composition NA, missing housing data"),
    # ---- card 4
    "BUY50": L("1=plans to buy or build (or already has);2=probably will;3=undecided, depends;4=will not;"
               "8=farm schedule;X=not ascertained"),
    "NEWOLD": L("1=new;2=old;8=farm schedule;9=don't know;0=will not buy;X=not ascertained;Y=NA whether will buy"),
    "RETYPE": L("1=other one- or two-family house;2=apartment house;3=summer or weekend home;"
                "4=commercial or rental property;5=farm or farm land;6=lot;7=other;8=three or more types;"
                "9=separate bare farm land (farm operators);0=none / no second type;X=type NA;Y=not ascertained whether"),
    "REWORTH": L("1=under $1,000;2=$1,000-4,999;3=$5,000-24,999;4=$25,000 and over;9=don't know;0=owns none;"
                 "X=value NA;Y=not ascertained whether owns"),
    "RE8": RE8,
    "SOLD": L("1=sold home only;2=sold other real estate only;3=sold both;4=sold neither;5=sold, NA which;"
              "0=sold neither (as punched);X=not ascertained"),
    "REOWED": bracket("1-199|200-499|500-999|1,000-1,999|2,000-2,999|3,000-4,999|5,000-9,999|10,000 or over"),
    "REMORT": bracket("1-499 (illegible)|illegible|500-999|illegible|2,000-2,999|3,000-4,999|5,000-9,999|10,000 or over"),
    "SIGN": L("1=positive;2=negative;0=zero / none;X=not ascertained;Y=not ascertained whether"),
    "REDEBTCLS": bracket("1-499|500-999|1,000-2,999|3,000-4,999|5,000-7,499|7,500-9,999|10,000 or over",
                         "8=owns, no debt;9=owns, NA whether debt;0=owns none;X=amount NA;Y=not ascertained whether owns"),
    "HPLAN": L("1=has bought / will buy new;2=has bought / will buy used;3=will buy, NA new/used;"
               "4=probably will buy new;5=probably will buy used;6=probably will buy, NA new/used;"
               "7=undecided - new;8=undecided - used;9=undecided, NA new/used;0=will not buy;"
               "Y=not ascertained;X=farm schedule, not asked"),
    # ---- card 5
    "SRC8": SRC8, "SRC9": SRC9,
    "INCCOMP": L("1=much larger in 1949 (+25% or more);2=somewhat larger (+5 to 24%);3=about the same;"
                 "4=somewhat smaller (-5 to 24%);5=much smaller (-25% or more);9=don't know;X=not ascertained"),
    "EXP50": L("1=larger in 1950;3=same;5=smaller in 1950;9=don't know;X=not ascertained"),
    "SECURE": L("1=very secure;2=rather secure;4=rather insecure;5=very insecure;9=don't know;X=not ascertained"),
    "SECFRAME": L("1=job, employment, business;2=financial resources, savings, debts;3=someone to depend on, relief;"
                  "4=expenses;5=age;6=health;7=ability or training;8=other personal factors;"
                  "9=factors affecting the country as a whole;Y=religious or philosophical;"
                  "0=no second frame of reference;X=not ascertained"),
    "JOBCHANCE": L("11=same pay: absolutely sure;12=same pay: could probably get;14=same pay: would be difficult;"
                   "15=same pay: impossible;21=less pay: absolutely sure;22=less pay: could probably get;"
                   "24=less pay: would be difficult;25=less pay: impossible;51=in a depression: absolutely sure;"
                   "52=in a depression: could probably get;54=in a depression: difficult;55=in a depression: impossible;"
                   "88=farm schedule;99=don't know;XX=not ascertained;00=inapplicable (self-employed, not working)"),
    "JOBREASON": L("11=shortage of workers in occupation;12=jobs plentiful;13=has had other job offers;"
                   "14=personal reputation or qualifications;15=through union or agency;16=by changing occupation;"
                   "17=has other skills;31=slack, jobs hard to get;32=shortage of work in occupation;"
                   "33=few jobs pay as well;34=personal qualifications, health;35=age, too old;36=no seniority on new job;"
                   "88=farm schedule;99=don't know;XX=not ascertained;00=inapplicable / no second reason"),
    "NRET": L("1=one return;2=two returns;3=three returns;4=four or more;0=no taxable income / under $600;X=not ascertained"),
    "NEXEMPT": L("0=none;1=one;2=two;3=three;4=four;5=five;6=six;7=seven;8=eight;9=nine or more;X=not ascertained"),
    "INCRATIO": L("1=less than 50%;2=50-74%;3=75-84%;4=85-94%;5=95-104%;6=105-114%;7=115-124%;8=125-149%;"
                  "9=150-199%;Y=200% or more;0=zero or negative income in either year;X=not ascertained"),
    "TAXCLS": bracket("1-49|50-99|100-199|200-499|500-999|1,000-1,999|2,000-4,999|5,000 and over",
                      "0=no income tax;X=not ascertained"),
    # ---- card 6
    "CAROWN": CAROWN, "CARNEW": CARNEW, "YEARS": YEARS,
    "MAKE": L("10=Chevrolet;12=Ford;13=Plymouth;16=other low-priced;20=Pontiac;21=Oldsmobile;22=Mercury;"
              "23=Dodge;24=De Soto;25=Kaiser;26=Nash;27=Hudson;28=Studebaker;29=other medium-priced;"
              "30=Cadillac;31=Buick;32=Lincoln;33=Chrysler;35=Frazer;36=Packard;37=LaSalle;38=other high-priced;"
              "99=don't know;00=no car;X0=make not ascertained;Y0=not ascertained whether owns"),
    "TRADED": L("1=traded in;2=sold;3=neither, but bought in 1949-50;0=did not buy in 1949-50;X=not ascertained"),
    "CARFIN": L("1=installment plan;2=borrowed some other way;3=own cash only (plus trade-in);"
                "4=installment and other borrowing;0=did not buy in 1949-50;X=not ascertained"),
    "CARMETH": L("1=cash, no trade-in, no borrowing;2=cash plus trade-in, no borrowing;"
                 "3=borrowing plus trade-in, borrowed under 50%;4=borrowing plus trade-in, borrowed 50% or more;"
                 "5=borrowing plus trade-in, percent NA;6=borrowing plus cash, borrowed under 50%;"
                 "7=borrowing plus cash, borrowed 50% or more;8=borrowing plus cash, percent NA;"
                 "0=did not buy in 1949-50;X=not ascertained"),
    "NEWUSED": L("1=new;2=used;0=no trade-in or sale;X=not ascertained;Y=not ascertained whether traded in"),
    "CARPLAN": CARPLAN, "HALF": HALF, "FINPLAN": FINPLAN, "DURTYPE": DURTYPE, "DUR8": DUR8,
    "CASHPROP": CASHPROP, "TRADEIN": TRADEIN,
    "DURFIN": L("1=all bought for cash;2=all bought on installment;3=some cash, some installment;0=none bought;"
                "X=not ascertained;Y=not ascertained whether bought"),
    "DURPLAN": L("1=definitely plans to buy (or has bought in 1950);2=probably will buy;3=depends, undecided;"
                 "0=does not plan to buy;X=not ascertained"),
    "DURPRICE": L("1=will go up a lot (11% or more);2=will go up somewhat;3=stay about the same;"
                  "4=will go down somewhat;5=will go down a lot;7='I hope they will come down';9=undecided, don't know;X=not ascertained"),
    "DURFACTOR": L("1=supply or demand for automobiles;2=supply or demand for other durables;3=supply or demand for both;"
                   "4=wages or costs of automobiles/durables;5=other explanation specific to durables;"
                   "6=general trend of all prices;7=general trend of wages and costs;8=other general explanation;"
                   "0=no explanation given;X=not ascertained"),
    # ---- card 7
    "MISCSAV": L("1=plus $1,000 or over;2=plus $500-999;3=plus $200-499;4=plus $1-199;5=minus $1-199;"
                 "6=minus $200-499;7=minus $500-999;8=minus $1,000 or over;0=zero;X=amount NA;Y=not ascertained"),
    "DEBTDIR": L("1=decrease in indebtedness (adds to saving);2=increase in indebtedness;0=no change;X=not ascertained"),
    "SAVSIGN": SAVSIGN, "SAVPCT": SAVPCT, "SAVCLS": SAVCLS, "B_SMALL6": B_SMALL6,
    "RESIDPCT": L("1=1-19%;2=20-39%;3=40-49%;4=50-59%;5=60-69%;6=70-79%;7=80-89%;8=90-99%;9=100% or over;"
                  "0=zero / no or negative disposable income;X=not ascertained"),
    "DEBTCHG": L("1=debts increased $1-199;2=debts increased $200-499;3=debts increased $500-999;"
                 "4=debts increased $1,000 or more;5=debts reduced $1-199;6=debts reduced $200-499;"
                 "7=debts reduced $500-999;8=debts reduced $1,000 or more;0=no change;X=not ascertained;Y=NA whether"),
    "BUSSAVCLS": L("1=$10,000 or more;2=$5,000-9,999;3=$3,000-4,999;4=$1,000-2,999;5=$500-999;6=$1-499;"
                   "7=minus $1-499;8=minus $500 or over;0=none;X=not ascertained;Y=NA whether"),
    "RESIDCLS": L("1=lowest bracket (bounds illegible in codebook);2=up to $999;3=$1,000-1,499;4=$1,500-1,999;"
                  "5=$2,000-2,999;6=$3,000-3,999;7=$4,000-4,999;8=$5,000-7,499;9=$7,500 or over;0=zero;X=not ascertained;Y=over $9,999 (see residual_expenditures)"),
    # ---- card 8
    "BUSACCT": L("1=separate (non-professional);2=not separate (non-professional);3=separate (professional);"
                 "4=not separate (professional);5=don't know (professional);9=don't know;"
                 "0=not a self-employed businessman;X=not ascertained;Y=NA whether self-employed"),
    "AFTRANS": L("1=bought bonds, none cashed or matured;2=cashed or matured, none bought;3=both bought and cashed;"
                 "4=owns bonds, no transactions;0=held none, no transactions;X=transactions NA;Y=not ascertained whether"),
    "AFCASHED": bracket("1-49|50-99|100-199|200-399|400-499|500-999|1,000-1,499|1,500-1,999|2,000 and over"),
    "LIQ8": LIQ8, "LIQ6": LIQ6, "LIQ5": LIQ5, "CHG9": CHG9,
    "LIQ7": bracket("1-199|200-499|500-999|1,000-1,999|2,000-4,999|5,000-9,999", "9=$10,000 and over;0=none;X=amount not ascertained;Y=not ascertained whether"),
    "LIQPCT": L("1=none a year ago, has some now;2=increase of 100% or more;3=increase 25-99%;4=increase 5-24%;"
                "5=about the same (96-104%);6=decrease 5-24%;7=decrease 25% or more;8=had assets a year ago, none now;"
                "0=none now and none a year ago;X=not ascertained;Y=not ascertained whether has/had"),
    "LIQCHG": L("1=increase $1,000 or more;2=increase $500-999;3=increase $200-499;4=increase $1-199;"
                "5=has assets, no change;6=decrease $1-199;7=decrease $200-499;8=decrease $500-999;"
                "Y=decrease $1,000 or more;0=none now and none a year ago;X=not ascertained"),
    "AFCHG": L("11=has bonds, no change;21=increase $2,000 or more;22=increase $1,000-1,999;23=increase $500-999;"
               "24=increase $200-499;25=increase $1-199;2X=increase, amount NA;26=accrual only, over $200;"
               "27=accrual only, $1-199;31=decrease $1-199;32=decrease $200-499;33=decrease $500-999;"
               "34=decrease $1,000-1,999;35=decrease $2,000 or more;3X=decrease, amount NA;"
               "00=none now, none a year ago;XX=has bonds, change NA;YY=not ascertained whether has/had"),
    "IBCHG": L("1=increase $10,000 or more;2=increase $5,000-9,999;3=increase $1,500-4,999;4=increase $1-1,499;"
               "5=has bonds, no change;6=decrease $1-1,499;7=decrease $1,500-4,999;8=decrease $5,000-9,999;"
               "9=decrease $10,000 or more;0=none now, none a year ago;X=change NA;Y=not ascertained whether"),
    "ACCTTYPE": L("1=bank only;2=non-bank savings accounts only;3=both;0=no savings accounts;Y=not ascertained"),
    "BANKBOOK": L("1=looked up bank book;2=asked, did not look up;3=not asked;4=NA whether asked, did not look up;"
                  "0=irrelevant, no bank account;X=not ascertained"),
    "NOBOOK": L("1=annoyed by request, refused;2=data not available at home;3=books hard to get out, in a hurry;"
                "4=sure of figures;5=other;0=inapplicable;X=not ascertained"),
    "LIQUSE": L("11=living expenses;12=hospital, medical, death, birth;21=repairs or additions to house;"
                "22=luxuries, travel;23=education;24=taxes;25=car or durable repairs;26=moving;31=car purchase;"
                "32=purchase of other durables;41=payment of debts;42=investment in real estate;"
                "43=investment in business;44=investment in securities;45=bought or built house to live in;"
                "46=other investment;47=loan to family;51=farm operating expense, machinery;52=purchased farm;"
                "61=gifts, annuities, trusts;77=other;00=none / did not reduce liquid assets;"
                "XX=purpose not ascertained;YY=not ascertained whether reduced"),
    # ---- card 9
    "B15": B15,
    "OWEDTO": L("1=hospital or doctor;2=lawyer;3=stock broker;4=overdue taxes (handwritten in codebook);"
                "5=other services (garage, building, etc.);6=family;8=miscellaneous debts incl. cemetery lots;9=uncertain, don't know;0=no such money owed;"
                "X=owed, NA to whom;Y=not ascertained whether owed"),
    "REPAID": L("1=repaid all or part of a debt;5=has debt, did not repay;9=don't know;0=no debt / no repayment;"
                "X=not ascertained whether repaid;Y=not ascertained whether had debt"),
    "NPOL": L("1=one policy;2=two policies;3=three policies;4=four or more;9=don't know;0=no life insurance;"
              "X=insured, number NA;Y=not ascertained whether insured"),
    "POLTYPE": L("11=one type: straight or limited-payment life;12=one type: can borrow on it;13=one type: endowment;"
                 "14=one type: type not ascertained;21=term insurance only;22=group insurance only;"
                 "23=one type: cannot borrow on it;24=burial insurance only;31=several types, can borrow on all;"
                 "32=several types, cannot borrow on any;33=several types, can borrow on some;34=several types, can borrow on one or more, NA on others;35=several types, NA whether can borrow on any;"
                 "88=answered in another frame of reference;99=don't know;00=no policy;XX=type not ascertained;"
                 "YY=not ascertained whether insured"),
    "POLCOUNT": L("1=one policy;2=two policies;3=three policies;4=four or more;5=has this type, number NA;"
                  "9=does not mention this type;0=question not asked (no insurance)"),
    "LUMP": bracket("1-199|200-499|500-999|1,000-2,999|3,000-4,999|5,000-9,999|10,000 or more"),
    "RETFUND": L("1=$1-49;2=$50-99 (inferred);3=$100-199;4=$200-499 (inferred);5=$500-999 (inferred);6=$1,000 or over;0=none;X=yes, amount NA;Y=not ascertained"),
    "STOCK": L("1=stock sold to the general public;2=stock in privately held corporation;3=both types;"
               "4=doesn't know type;0=does not own stock;X=type not ascertained;Y=not ascertained whether owns"),
    "STOCKSIZE": L("1=under $500;2=$500-999;3=$1,000-4,999;4=$5,000-24,999;5=$25,000-99,999;6=$100,000 or more;"
                   "0=does not own;X=amount NA;Y=ownership not ascertained"),
    "STOCKTRADE": L("1=bought;2=sold;3=bought and sold, bought more;4=bought and sold, sold more;"
                    "5=bought and sold, NA which was larger;6=bought and sold, equal amounts;"
                    "0=neither bought nor sold;X=not ascertained"),
    "BUSMONEY": bracket("1-499|500-999|1,000-2,999|3,000-4,999|5,000-9,999|10,000 or more"),
    "BUSWORTH": L("1=under $1,000;2=$1,000-4,999;3=$5,000-24,999;4=$25,000-99,999;5=$100,000 or more;9=don't know;"
                  "0=no business now;X=worth not ascertained;Y=not ascertained whether has business"),
    "OWED": L("1=owed money, lent before 1949;2=owed money, lent in 1949;3=owed money, lent both periods;"
              "9=don't know;0=not owed any money;X=not ascertained;Y=not ascertained whether owed"),
    "LENT": bracket("1-499|500-999|1,000-1,999|2,000-4,999|5,000 and more"),
    "INHERIT": bracket("1-299|300-499|500-999|1,000-4,999|5,000 or over"),
    "BIGEXP": L("11=moving;12=additions or repairs to another house;21=major hospital or medical expenses;"
                "22=minor hospital or medical expenses;23=hospital or medical, size NA;"
                "31=change in SU composition;32=establishment of new household;33=support of other families;"
                "41=jewelry, furs;42=other luxury (travel, vacations);51=auto repair;61=gifts and contributions;"
                "62=education;71=marriage expenses;88=other large expenses;99=don't know;00=none / no second type;XX=not ascertained"),
    "SPEND": L("1=much more (+25% or more);2=more (+5 to 24%);3=about the same;4=less (-5 to 24%);"
               "5=much less (-25% or more);9=don't know;X=not ascertained"),
    "SPENDWHY": L("1=more: higher prices;2=more: more people in unit;3=more: has moved, rent raised;"
                  "4=more: needed more things, income higher;5=less: lower prices;6=less: fewer people in unit;"
                  "7=less: has moved, rent lowered;8=less: needed fewer things;9=less: economized;"
                  "0=no change or no reason;X=not ascertained"),
    "INSCLS": L("1=$1-49;2=$50-99;3=$100-199;4=$200-499;5=$500-999;6=$1,000 or over;8=insured, paid no premiums;9=insured, NA whether paid premiums;"
                "0=no life insurance;X=amount not ascertained;Y=not ascertained whether insured"),
    "DEBT8": DEBT8,
    # ---- card Y
    "FAMTYPE": L("0=single-SU family, primary, no secondaries;1=single-SU family with unrelated secondary;"
                 "2=single-SU family, unrelated secondary;3=family of 2 SUs, all complete;"
                 "4=family of 3+ SUs, all complete;5=two-SU family, matched;6=three-SU family, matched;"
                 "9=SU composition NA / no family record"),
    "FAMCHG": L("1=single-SU family: increase or decrease of adults;2=single-SU family: no change;"
                "X=single-SU family: not ascertained;3=increase of one adult;4=increase of two or more adults;"
                "5=increase, number NA;6=decrease of one adult;7=decrease of two or more adults;"
                "8=decrease, number NA;0=multiple-SU family: no change;Y=not ascertained / no family record"),
    "DURCLS": bracket("1-99|100-199|200-299|300-499|500-749|750-999|1,000-1,499|1,500-1,999|2,000 or over",
                      "0=zero;X=not ascertained whether bought;Y=bought, amount NA"),
    "FAMSAV": L("1=single-SU family: positive savings;2=single-SU family: positive from bond accrual only;"
                "3=single-SU family: negative savings;0=single-SU family: zero savings;X=sign not ascertained;"
                "4=all SUs positive savers;5=all SUs zero savers;6=all SUs negative savers;"
                "7=mixed: positive and negative (or zero);8=mixed: positive and zero;9=mixed: negative and zero;"
                "Y=sign NA for one or more SUs / no family record"),
    "FILTER_FAM": FILTER_FAM,
}

# --------------------------------------------------------------------------- layout
# card | columns | name | kind | labels | description
# kinds: c = categorical code     n1/n10/n100[:meaning of trailing Y] = dollars in $1/$10/$100 units
#        pct = two-digit percent  w = weight                           s = string kept as punched
LAYOUT = """
1|8|interview_result|c|-|Result of interview (1 = interview; all 3,512 cases)
1|9-12|sample_book_number|s|-|Sample book number; first two digits identify the sample point
1|13|sampling_ratio|c|RATIO|Sampling ratio / schedule type
1|14|place_code|c|PLACE10|Place code (size of place by South / not South)
1|15-16|su_composition|c|SUCOMP|Spending unit composition of the dwelling unit
1|17-19|interviewer_code|s|-|Interviewer code
1|20|interview_date|c|DATE|Date of interview
1|21|respondent_is_head|c|RHEAD|Is respondent the head of the spending unit?
1|22|adults_in_su|c|N4|Adults in spending unit (including wife)
1|23|why_not_separate_su_1|c|WHYNOTSU|Why other adult is not a separate spending unit (1st response)
1|24|why_not_separate_su_2|c|WHYNOTSU|Why other adult is not a separate spending unit (2nd response)
1|25|su_composition_change|c|SUCHANGE|Change in adults in spending unit since a year ago
1|26|children_under_18|c|CHILD|Children under 18 in spending unit
1|27|people_in_su|c|PEOPLE|People in spending unit (adults + children)
1|28|child_born_1949_50|c|BORN|Any children born during 1949 or 1950?
1|29|outside_dependents|c|DEPEND|Dependent relatives living elsewhere
1|30|years_married|c|MARRIED|How long married
1|31|age_head|c|AGE|Age of head of spending unit
1|32|education_head|c|EDUC|Education of head
1|33|wwii_veteran|c|VET|World War II veteran status of spending unit
1|34|sex_head|c|SEX|Sex of head of spending unit
1|35|sex_respondent|c|SEX|Sex of respondent
1|36|race|c|RACE|Race (as recorded by interviewer)
1|37|call_number|c|CALLS|Number of calls needed to obtain interview
1|38-39|occupation_head|c|OCC|Occupation of head of spending unit
1|40|self_employed|c|SELFEMP|Works for self or someone else
1|41|months_worked_1949|c|MONTHS|How much of 1949 head worked
1|42|owns_business|c|OWNBUS|Owns a business or part of one (now or in 1949)
1|43-44|business_type|c|BUSTYPE|Kind of business
1|45|business_incorporated|c|CORP|Corporation or unincorporated business
1|46|place_size|c|PLACE6|Place code, six size classes
1|47|income_1949_class_card1|c|INC8|1949 income class (copy of card 5 col 63)
1|48|occupation_group|c|OCC10|Special occupation code built by SRC from cols 38-39
1|63-64|family_composition_code|s|-|Special family code: 1st digit = years-married class (col 30), 2nd digit = number of children (X = NA); 00 not married, 01 not married with children
1|73|family_status|c|FAMSTAT|Family life-cycle status
1|74-75|sampling_split|c|SPLIT|Sampling split (20-73 = first two digits of sample book number)
1|76|undocumented_c1_76|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
1|79-80|weight|w|-|Spending-unit weight (card 1). USE THIS for spending-unit estimates
2|8|fin_position_vs_year_ago|c|BETTER3|Better or worse off financially than a year ago
2|9|earning_rate_vs_year_ago|c|MORE3|Making as much money now as a year ago
2|10-11|fin_change_reason_1|c|REASON_FIN|Cause of changed financial position (1st)
2|12-13|fin_change_reason_2|c|REASON_FIN|Cause of changed financial position (2nd)
2|14|income_expected_year_ahead|c|WILL3|Expect to be making more or less a year from now
2|15-16|income_expect_reason_1|c|REASON_EXP|Reason for expected income change (1st)
2|17-18|income_expect_reason_2|c|REASON_EXP|Reason for expected income change (2nd)
2|19|times_now|c|TIMES5|Country having good or bad times now
2|20-21|times_now_reason_1|c|REASON_TIMES|Factor making for good or bad times now (1st)
2|22-23|times_now_reason_2|c|REASON_TIMES|Factor making for good or bad times now (2nd)
2|24|times_vs_year_ago|c|TIMESCHG5|Times better or worse than a year ago
2|25-26|times_change_reason_1|c|REASON_TIMES|Factor making times better or worse (1st)
2|27-28|times_change_reason_2|c|REASON_TIMES|Factor making times better or worse (2nd)
2|29|times_next_12_months|c|TIMES5|Good or bad times in next twelve months
2|30-31|times_next_reason_1|c|REASON_TIMES|What respondent sees for 1950 (1st)
2|32-33|times_next_reason_2|c|REASON_TIMES|What respondent sees for 1950 (2nd)
2|34|prices_expected_1950|c|PRICES|What will happen to prices during 1950
2|35-36|price_reason_1|c|REASON_PRICE|Factor affecting prices (1st)
2|37-38|price_reason_2|c|REASON_PRICE|Factor affecting prices (2nd)
2|39|price_change_size|c|PRICEAMT|How much higher or lower prices by end of 1950
3|8|housing_tenure|c|TENURE|Own home or pay rent
3|9|neither_own_nor_rent|c|NEITHER|If neither owns nor rents: how is that
3|10-12|monthly_rent|n1|-|Monthly rent, dollars
3|13|rent_furnished|c|FURN|Rented furnished or unfurnished
3|14|rent_includes_1|c|RENTINCL|What rent includes (1st)
3|15|rent_includes_2|c|RENTINCL|What rent includes (2nd)
3|16|rents_out_part|c|RENTOUT|Rents out part of building
3|17|house_year_bought|c|BOUGHT|When home was bought
3|18|house_new_when_bought|c|NEWBUILT|Newly built or lived in before
3|19|house_year_built|c|BUILT|When house was built
3|20-22|house_value|n100|-|Estimated present value of house (punched in $100s)
3|23|house_value_certainty|c|CERTAIN|Degree of certainty about current value
3|24-27|house_debt_payments_1949|n1|-|Total house debt payments in 1949
3|28-31|house_principal_payments_1949|n1|-|Payment on principal only in 1949
3|32|knows_principal_payment|c|KNOWPRIN|Does R know amount paid on principal
3|33|house_paid_before_1949|c|PRE49|Payments before 1949 on house bought 1949-50 (class)
3|34|house_payments_1949_class|c|HOUSE9|House payments in 1949 on 1949 purchase (class)
3|35|house_share_paid_before_1949|c|PROPPAID|Proportion paid before 1949
3|36|house_purchase_debt|c|HDEBT49|Debt incurred for 1949-50 house purchase
3|37|house_funds_source_1|c|FUNDS|Source of funds for 1949-50 house purchase (1st)
3|38|house_funds_source_2|c|FUNDS|Source of funds for 1949-50 house purchase (2nd)
3|39|repairs_type_1|c|REPAIRS|Type of additions and repairs (1st)
3|40|repairs_type_2|c|REPAIRS|Type of additions and repairs (2nd)
3|41-42|house_additions_spending|n100|-|Expenditures for additions (punched in $100s)
3|43-44|house_repairs_spending|n100|-|Expenditures for repairs (punched in $100s)
3|45|house_unclassifiable_spending|c|UNCLASS|Unclassifiable housing expenditures (class)
3|46-49|house_upkeep_total|n1|-|Total expenditures for additions, repairs and unclassifiable
3|50|repairs_debt_class|c|REPDEBT|Present debt on repairs or additions (class)
3|51|house_debt_pct_value|c|DEBTPCT|House debt as percent of house value
3|52-54|house_debt|n100|-|House debt at present (punched in $100s)
3|55|house_equity_class|c|HOUSE9|Net equity in owned house (class)
3|56|principal_pct_of_payment|c|PRINPCT|Principal as percent of total house debt payment
3|57|upkeep_pct_value|c|PCTVAL|Additions and repairs as percent of house value
3|58|additions_pct_value|c|PCTVAL|Additions only as percent of house value
3|59|rent_pct_income|c|RENTPCT|Rent as percent of disposable income
3|60|monthly_rent_class|c|RENTCLS|Monthly rent (class)
3|61|house_value_class|c|HOUSE9|Value of owned home (class)
3|62|house_debt_class|c|HDEBTCLS|House debt at present (class)
3|63|house_debt_payments_class|c|PAY7|Total house debt payments in 1949 (class)
3|64|house_principal_class|c|PAY7|Payment on principal in 1949 (class)
3|65|undocumented_c3_65|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
3|71|undocumented_c3_71|c|-|UNDOCUMENTED column (not in codebook)
3|72|house_purchase_1949_50|c|HPURCH|1949 and 1950 actual nonfarm house purchase
3|73|upkeep_type|c|UPKEEP|Type of 1949 upkeep expenditure on nonfarm home
3|78|housing_filter|c|FILTER_H|Filter: status of housing data
3|79-80|weight_housing|w|-|Housing weight (card 3): 0 for farm and related-secondary units
4|8|house_plan_1950|c|BUY50|Expect to buy or build a house in 1950
4|9|house_plan_1951|c|BUY50|Expect to buy or build a house in 1951
4|10|house_plan_new_or_old|c|NEWOLD|Planned house newly built or lived in before
4|11-13|house_plan_price|n100:farm_schedule|-|Planned purchase price (punched in $100s)
4|14|other_real_estate_1|c|RETYPE|Other real estate owned (1st type)
4|15|other_real_estate_2|c|RETYPE|Other real estate owned (2nd type)
4|16|other_real_estate_worth|c|REWORTH|Value of other real estate (class)
4|17-19|other_real_estate_debt|n100|-|Debt on other real estate (punched in $100s)
4|20|other_re_principal_class|c|PAY7|Payments on principal, other real estate (class)
4|21|other_re_bought_1949_class|c|RE8|Price of other real estate bought in 1949 (class)
4|22|sold_real_estate_1949|c|SOLD|Sold home or other real estate in 1949
4|23|real_estate_sold_type|c|RETYPE|What other real estate was sold
4|24|real_estate_sold_price_class|c|RE8|Price of real estate sold (class)
4|25|owed_to_r_on_sale_class|c|REOWED|Amount still owed to R on 1949 sale (class)
4|26|mortgage_at_sale_class|c|REMORT|Mortgage due at time of sale (class)
4|27|re_saving_sign|c|SIGN|Sign of net saving from other real estate transactions
4|28-30|re_saving_amount|n100|-|Net saving from other real estate transactions, unsigned (punched in $100s)
4|31|house_plan_price_class|c|HOUSE9|Planned house purchase price (class)
4|32|other_re_debt_class|c|REDEBTCLS|Debt on other real estate (class)
4|35|house_plan_1950_detail|c|HPLAN|1950 nonfarm house purchase plans
4|36|house_plan_1951_detail|c|HPLAN|1951 nonfarm house purchase plans
4|37|house_plan_1951_price_class|c|HOUSE9|Price of expected 1951 house purchase (class)
5|8-10|income_wages|n100|-|Wage and salary income of SU, 1949 (punched in $100s)
5|11|income_roomers_class|c|SRC8|Income from roomers and boarders (class)
5|12|income_rent_class|c|SRC8|Income from other rent (class)
5|13|income_interest_dividends_class|c|SRC8|Income from interest, dividends, trusts, royalties (class)
5|14-16|income_unincorp_business|n100:loss|-|Income from unincorporated business (punched in $100s)
5|17|business_profit_left_in_class|c|SRC8|Profit left in the business (class)
5|18|business_loss_class|c|SRC8|Business loss (class)
5|19|income_professional_farm_class|c|SRC8|Income from professional practice, self-employment, farming (class)
5|20|income_transfers_class|c|SRC8|Income from pensions, allotments, unemployment compensation, welfare, alimony (class)
5|21|income_receivers|c|N4|Number of income receivers in spending unit
5|22|income_1949_assigned|c|ASSIGNED|Was total 1949 income assigned
5|23-27|income_1949|n1:negative|-|TOTAL 1949 MONEY INCOME before taxes, dollars
5|28-32|income_1948|n1:negative|-|Total 1948 income, dollars (only when stated by respondent)
5|33|income_1949_vs_1948|c|INCCOMP|1949 income compared with 1948
5|34|income_1948_class|c|INC8|1948 income (class; code 1 = $1-999)
5|35|income_expected_1950|c|EXP50|Expected 1950 income compared with 1949
5|36|financial_security|c|SECURE|How secure R feels financially
5|37|security_frame_1|c|SECFRAME|Frame of reference for security evaluation (1st)
5|38|security_frame_2|c|SECFRAME|Frame of reference for security evaluation (2nd)
5|39-40|job_chances|c|JOBCHANCE|Chances of getting another job at same pay if job lost
5|41-42|job_chances_reason_1|c|JOBREASON|Reason another job would be easy or hard to get (1st)
5|43-44|job_chances_reason_2|c|JOBREASON|Reason another job would be easy or hard to get (2nd)
5|45|disposable_income_assigned|c|ASSIGNED|Was 1949 disposable income assigned
5|46-50|disposable_income_1949|n1:negative|-|1949 disposable income (income less federal income tax)
5|51|income_tax_assigned|c|ASSIGNED|Was 1949 income tax assigned
5|52-56|income_tax_1949|n1|-|1949 federal income tax (estimated by SRC)
5|57|tax_returns|c|NRET|Number of tax returns for SU
5|58|tax_exemptions|c|NEXEMPT|Number of tax exemptions for SU
5|59-60|tax_pct_income|pct|-|Taxes as percent of income
5|61|income_1949_pct_1948_class|c|INCRATIO|1949 income as percent of 1948 (class)
5|62|disposable_income_class|c|INC8|1949 disposable income (class)
5|63|income_1949_class|c|INC8|1949 income (class)
5|64|income_tax_class|c|TAXCLS|1949 income tax (class)
5|65|income_unincorp_class|c|SRC9|Income from unincorporated business (class)
5|66|income_wages_class|c|SRC9|Income from wages and salaries (class)
5|67|undocumented_c5_67|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
6|8|car_ownership|c|CAROWN|Owns a car
6|9|car_bought_new_or_used|c|CARNEW|Car bought new or used
6|10-11|car_model_year_code|c|YEARS|Year model of car
6|12-13|car_make|c|MAKE|Make of car
6|14-15|car_year_bought_code|c|YEARS|Year car was bought
6|16-17|car_price|n100|-|Total price of car bought 1949-50 (punched in $100s)
6|18|car_trade_in|c|TRADED|Traded in or sold a car in 1949-50
6|19|car_financing|c|CARFIN|How car purchase was financed
6|20-21|car_down_payment|n100|-|Cash down payment on car (punched in $100s)
6|22|car_financing_method|c|CARMETH|Method of financing car purchase
6|23-24|trade_in_model_year_code|c|YEARS|Year model of car traded in or sold
6|25-26|trade_in_year_bought_code|c|YEARS|Year of purchase of car traded in or sold
6|27|trade_in_new_or_used|c|NEWUSED|Trade-in was new or used when bought
6|28|car_plan|c|CARPLAN|Expects to buy a car in 1950
6|29|car_plan_half|c|HALF|Expects to buy car in first or second half of year
6|30-31|car_plan_price|n100|-|Expected price of car (punched in $100s)
6|32|car_plan_financing|c|FINPLAN|How expected car purchase will be financed
6|33|durables_bought_1|c|DURTYPE|Durable goods bought in 1949 (1st)
6|34|durables_bought_2|c|DURTYPE|Durable goods bought in 1949 (2nd)
6|35|refrigerator_price_class|c|DUR8|Price of refrigerator bought in 1949 (class)
6|36|refrigerator_cash_share|c|CASHPROP|Share of refrigerator price paid in cash
6|37|refrigerator_trade_in|c|TRADEIN|Trade-in on refrigerator
6|38|radio_price_class|c|DUR8|Price of radio bought in 1949 (class)
6|39|radio_cash_share|c|CASHPROP|Share of radio price paid in cash
6|40|radio_trade_in|c|TRADEIN|Trade-in on radio
6|41|tv_price_class|c|DUR8|Price of television set bought in 1949 (class)
6|42|tv_cash_share|c|CASHPROP|Share of television price paid in cash
6|43|tv_trade_in|c|TRADEIN|Trade-in on television set
6|44|furniture_price_class|c|DUR8|Price of furniture bought in 1949 (class)
6|45|furniture_cash_share|c|CASHPROP|Share of furniture price paid in cash
6|46|washer_price_class|c|DUR8|Price of washing machine bought in 1949 (class)
6|47|washer_cash_share|c|CASHPROP|Share of washing machine price paid in cash
6|48-51|durables_spending_1949|n1|-|Total price of durable goods bought in 1949, excluding car
6|52|durables_financing|c|DURFIN|Method of financing 1949 durable purchases
6|53|durables_plan_1950|c|DURPLAN|Expects to buy durables in 1950
6|54|durables_plan_item_1|c|DURTYPE|Expected durable purchase (1st)
6|55|durables_plan_item_2|c|DURTYPE|Expected durable purchase (2nd)
6|56-59|durables_plan_spending|n1|-|Expected spending on durables in 1950
6|60|refrigerator_plan_class|c|DUR8|Expected spending on refrigerator (class)
6|61|radio_plan_class|c|DUR8|Expected spending on radio (class)
6|62|tv_plan_class|c|DUR8|Expected spending on television set (class)
6|63|furniture_plan_class|c|DUR8|Expected spending on furniture (class)
6|64|washer_plan_class|c|DUR8|Expected spending on washing machine (class)
6|65|refrigerator_plan_half|c|HALF|Expects to buy refrigerator in first or second half
6|66|tv_plan_half|c|HALF|Expects to buy television in first or second half
6|67|durable_prices_expected|c|DURPRICE|Expected change in car and durable prices
6|68|durable_prices_reason|c|DURFACTOR|Factor influencing durable goods prices
6|69-70|car_net_price|n100|-|Price of car minus trade-in or sale (punched in $100s)
6|71-74|durables_net_spending|n1|-|Net consumer durable spending incl. car, after trade-ins
7|8|misc_saving_class|c|MISCSAV|Miscellaneous saving (class)
7|9-12|installment_debt_increase|n1|-|Increase in installment debt from 1949 purchases
7|13|consumer_debt_change_dir|c|DEBTDIR|Direction of change in total consumer indebtedness
7|14-17|consumer_debt_change_amount|n1|-|Change in total consumer indebtedness, unsigned
7|18|business_saving_sign|c|SIGN|Sign of business saving
7|19-22|business_saving_amount|n1|-|Business saving, unsigned
7|23-26|contractual_saving|n1|-|Contractual saving (always positive)
7|27|saving_sign|c|SAVSIGN|Sign of total saving (Savings A)
7|28|saving_assigned|c|ASSIGNED|Was saving amount assigned
7|29-33|saving_amount|n1|-|Amount of saving or dissaving (Savings A), unsigned
7|34-35|saving_pct_income|pct|-|Percent of income saved or dissaved (sign in saving_sign)
7|36-39|residual_expenditures|n1|-|Residual expenditures (disposable income less saving and other accounted items)
7|40|residual_pct_income_class|c|RESIDPCT|Residual expenditures as percent of disposable income
7|41|saving_pct_disposable_class|c|SAVPCT|Saving as percent of disposable income (class)
7|42-43|life_insurance_pct_income|pct|-|Life insurance premiums as percent of disposable income
7|44|installment_debt_increase_class|c|B_SMALL6|Increase in installment debt (class)
7|45|consumer_debt_change_class|c|DEBTCHG|Change in consumer indebtedness (class)
7|46|business_saving_class|c|BUSSAVCLS|Business saving (class)
7|47|contractual_saving_class|c|B_SMALL6|Contractual saving (class)
7|48|saving_class|c|SAVCLS|Total Savings A (class)
7|49|saving_pct_income_class|c|SAVPCT|Savings A as percent of income (class)
7|50|residual_expenditures_class|c|RESIDCLS|Residual expenditures (class)
7|51|undocumented_c7_51|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
7|52|undocumented_c7_52|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
7|53|undocumented_c7_53|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
7|54|undocumented_c7_54|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
8|8|business_accounts_separate|c|BUSACCT|Business and personal accounts kept separate
8|9|af_bonds_assigned|c|ASSIGNED|Was A-F bond amount assigned
8|10-14|af_bonds|n1|-|Present holdings of Series A-F savings bonds
8|15|af_bond_transactions|c|AFTRANS|A-F bond transactions in 1949
8|16|af_bonds_cashed_class|c|AFCASHED|A-F bonds matured or cashed in 1949 (class)
8|17|af_bonds_year_ago_class|c|LIQ6|A-F bonds a year ago (class)
8|18-19|af_bonds_change|c|AFCHG|Change in A-F bond holdings since a year ago
8|20|other_bonds_assigned|c|ASSIGNED|Was interest-paying bond amount assigned
8|21-23|other_govt_bonds|n100|-|U.S. bonds paying interest currently (punched in $100s)
8|24|all_bonds_class|c|LIQ6|Total of all bond holdings (class)
8|25|other_bonds_change|c|IBCHG|Change in interest-paying bonds since a year ago
8|26|savings_account_types|c|ACCTTYPE|Types of savings accounts held
8|27|savings_accounts_assigned|c|ASSIGNED|Was savings account amount assigned
8|28-32|savings_accounts|n1|-|Present holdings in savings accounts (bank, postal, S&L, credit union)
8|33|savings_accounts_year_ago_class|c|LIQ5|Savings accounts a year ago (class)
8|34|savings_accounts_change|c|CHG9|Change in savings accounts since a year ago
8|35|checking_assigned|c|ASSIGNED|Was checking account amount assigned
8|36-40|checking_account|n1|-|Present holdings in checking account
8|41|checking_year_ago_class|c|LIQ5|Checking account a year ago (class)
8|42|checking_change|c|CHG9|Change in checking account since a year ago
8|43|liquid_assets_year_ago_class|c|LIQ8|Total liquid assets a year ago (class)
8|44|liquid_assets_assigned|c|ASSIGNED|Was liquid asset amount assigned
8|45-49|liquid_assets|n1|-|TOTAL LIQUID ASSETS at time of interview (bonds + savings + checking)
8|50|liquid_assets_change_sign|c|SAVSIGN|Sign of change in liquid assets
8|51-55|liquid_assets_change_amount|n1|-|Change in liquid assets since a year ago, unsigned
8|56|bank_book_report|c|BANKBOOK|Did respondent look up bank book (codebook col 57)
8|57|bank_book_refusal_reason|c|NOBOOK|Reason for not looking up bank book (codebook col 58)
8|58-59|liquid_assets_use_1|c|LIQUSE|Purpose for which liquid assets were used in 1949, 1st (codebook cols 59-60)
8|60-61|liquid_assets_use_2|c|LIQUSE|Purpose for which liquid assets were used in 1949, 2nd (codebook cols 61-62)
8|62|liquid_assets_pct_change|c|LIQPCT|Percentage change in liquid assets (codebook col 63; labels verified against dollars)
8|63|af_bonds_class|c|LIQ6|A-F bonds (class) (codebook col 64)
8|64|other_bonds_class|c|LIQ6|Interest-paying bonds (class) (codebook col 65)
8|65|savings_accounts_class|c|LIQ6|Savings accounts (class) (codebook col 67)
8|66|checking_class_assigned|c|ASSIGNED|Was checking account class assigned (matches col 35)
8|67|checking_class|c|LIQ6|Checking account (class) (codebook col 69)
8|68|liquid_class_assigned|c|ASSIGNED|Was liquid asset class assigned (9 punched for all holdings of $10,000+)
8|69|liquid_assets_class|c|LIQ7|Total liquid assets (class) (codebook col 71; top class punched 9 = $10,000 and over)
8|70|undocumented_c8_70|c|-|UNDOCUMENTED: punched 5/9
8|71|liquid_assets_change_class|c|LIQCHG|UNDOCUMENTED in codebook: change in liquid assets (class), identified from the dollar change
8|72|undocumented_c8_72|c|-|UNDOCUMENTED: punched 1/5, a copy of the savings-account assigned flag (col 27)
9|8|installment_debt_nondurables|c|YN|Installment debt on non-durables
9|9|installment_debt_pre_1949|c|YN|Installment debt still owed from before 1949
9|10|owes_on_loans|c|YN|Owes on loans from bank, loan company or individual
9|11|owes_on_insurance_loan|c|YN|Owes on life insurance policy loan
9|12-14|personal_debt|n100|-|Total owed on purchases or loans (punched in $100s)
9|15|borrowed_1949_class|c|B15|Amount of this borrowed during 1949 (class)
9|16|other_debt_type|c|OWEDTO|Other money owed at end of 1949 (to whom)
9|17-19|other_debt|n100|-|Amount of other money owed (punched in $100s)
9|20|other_debt_borrowed_1949_class|c|B15|Other money borrowed in 1949 (class)
9|21|repaid_debt_1949|c|REPAID|Repaid all or part of any debt in 1949
9|22|other_debt_repaid_class|c|B15|Net repayment on other debt in 1949 (class)
9|23|life_insurance_policies|c|NPOL|Carries life insurance: number of policies
9|24-25|life_insurance_type|c|POLTYPE|Type of policy and whether can borrow on it
9|26|straight_life_policies|c|POLCOUNT|Straight life policies
9|27|limited_payment_policies|c|POLCOUNT|Limited-payment life policies
9|28|endowment_policies|c|POLCOUNT|Endowment policies
9|29|term_policies|c|POLCOUNT|Term policies
9|30|group_policies|c|POLCOUNT|Group policies
9|31|burial_policies|c|POLCOUNT|Burial insurance
9|32-35|life_insurance_premiums|n1|-|Life insurance premiums paid in 1949
9|36-37|life_insurance_face_value_code|s|-|Face value of life insurance (two-digit class code; see codebook card IX p.7-8)
9|38|insurance_lump_sum_class|c|LUMP|Lump-sum insurance payment received (class)
9|39|retirement_fund_class|c|RETFUND|Income put into retirement or pension fund (class; codes 2, 4, 5 illegible)
9|40|owns_stock|c|STOCK|Owns common or preferred stock
9|41|stock_value_class|c|STOCKSIZE|Value of publicly traded stock owned (class)
9|42|stock_trading_1949|c|STOCKTRADE|Bought or sold stocks or non-government bonds in 1949 
9|43-44|stock_net_purchase_amount|n100|-|Net purchases or sales of securities, unsigned (punched in $100s)
9|45|business_money_withdrawn_class|c|BUSMONEY|Money withdrawn from business investment (class)
9|46|business_money_invested_class|c|BUSMONEY|New money invested in business (class)
9|47|business_worth_class|c|BUSWORTH|Worth of interest in business (class)
9|48|owed_money_by_others|c|OWED|Does anybody owe R money
9|49|amount_lent_1949_class|c|LENT|Amount lent in 1949 (class)
9|50|loan_repayments_received_class|c|LENT|Repayment received on loans (class)
9|51|inheritance_gifts_class|c|INHERIT|Inheritance or large gifts received in 1949 (class)
9|52-53|large_expense_1|c|BIGEXP|Large expenses not otherwise covered (1st)
9|54-55|large_expense_2|c|BIGEXP|Large expenses not otherwise covered (2nd)
9|56|regular_spending_vs_year_ago|c|SPEND|Spending more or less on regular expenses than a year ago
9|57|spending_change_reason|c|SPENDWHY|Reason for spending more or less
9|58-61|total_debt|n100|-|TOTAL DEBT of spending unit (punched in $100s)
9|62|life_insurance_premium_class|c|INSCLS|Life insurance premiums (class)
9|63|total_debt_class|c|DEBT8|Total debt (class)
9|64|personal_debt_class|c|DEBT8|Owed on purchases or loans (class)
9|65|other_debt_class|c|DEBT8|Other money owed (class)
Y|8|fam_place_code|c|PLACE10|FAMILY CARD: place code
Y|9|fam_su_structure|c|FAMTYPE|Number and type of spending units in family
Y|10|fam_children_under_18|c|CHILD|Children under 18 in family
Y|11|fam_people|c|PEOPLE|People in family
Y|12|fam_child_born_1949_50|c|BORN|Any children in family born 1949-50
Y|13|fam_outside_dependents|c|DEPEND|Dependents not living with family
Y|14-15|fam_occupation_head|c|OCC|Occupation of head of primary spending unit
Y|16|fam_age_head|c|AGE|Age of head of primary spending unit
Y|17|fam_education_head|c|EDUC|Education of head of primary spending unit
Y|18|fam_wwii_veteran|c|VET|WWII veteran status, all SUs in family
Y|19|fam_income_receivers|c|N4|Income receivers in family
Y|20|fam_car_ownership|c|CAROWN|Family owns a car
Y|21|fam_car_new_or_used|c|CARNEW|Family car bought new or used
Y|22-23|fam_car_model_year_code|c|YEARS|Latest year model owned by family
Y|24-25|fam_car_year_bought_code|c|YEARS|Year latest model was bought
Y|26|fam_car_plan|c|CARPLAN|Family expects to buy a car
Y|27|fam_composition_change|c|FAMCHG|Change in family composition
Y|28|fam_income_assigned|c|ASSIGNED|Was family income assigned
Y|29-32|fam_income_1949|n10:negative|-|Total 1949 FAMILY income (punched in $10s)
Y|33|fam_income_class|c|INC8|1949 family income (class)
Y|34|fam_liquid_assets_assigned|c|ASSIGNED|Was family liquid asset amount assigned
Y|35-38|fam_liquid_assets|n10|-|Family liquid assets (punched in $10s)
Y|39|fam_liquid_assets_class|c|LIQ8|Family liquid assets (class)
Y|40|fam_liquid_year_ago_assigned|c|ASSIGNED|Was year-ago family liquid asset amount assigned
Y|41-44|fam_liquid_assets_year_ago|n10|-|Family liquid assets a year ago: FIELD WAS NEVER PUNCHED (all zero or NA)
Y|45|fam_liquid_year_ago_class|c|LIQ8|Family liquid assets a year ago (class)
Y|46|fam_disposable_income_assigned|c|ASSIGNED|Was family disposable income assigned
Y|47-50|fam_disposable_income_1949|n10:negative|-|1949 family disposable income (punched in $10s)
Y|51|fam_disposable_class_assigned|c|ASSIGNED|Was family disposable income class assigned
Y|52|fam_disposable_income_class|c|INC8|Family disposable income (class)
Y|53|fam_tax_class_assigned|c|ASSIGNED|Was family income tax class assigned
Y|54|fam_income_tax_class|c|TAXCLS|Family 1949 income tax (class)
Y|55-56|fam_tax_pct_income|pct|-|Family taxes as percent of income
Y|57|fam_tax_returns|c|NRET|Number of tax returns for family
Y|58|fam_tax_exemptions|c|NEXEMPT|Number of tax exemptions for family
Y|59-61|fam_durables_net_spending|n10|-|Family net durable goods spending incl. car (punched in $10s)
Y|62|fam_durables_class_unreliable|c|-|DO NOT USE: codebook says durable spending class, but codes 1-6 do not match the dollar amount in cols 59-61
Y|63|fam_saving_pattern|c|FAMSAV|Saving behaviour of spending units in family
Y|64|fam_saving_sign|c|SAVSIGN|Sign of family saving (Savings A)
Y|65|fam_saving_assigned|c|ASSIGNED|Was family saving assigned
Y|66-69|fam_saving_amount|n10|-|Family saving or dissaving, unsigned (punched in $10s)
Y|70|fam_saving_class|c|SAVCLS|Family Savings A (class)
Y|71|fam_rent_pct_income|c|RENTPCT|Rent as percent of family disposable income
Y|72|fam_saving_pct_income_class|c|SAVPCT|Family Savings A as percent of income (class)
Y|73|fam_income_decile_code|c|-|Family income decile as punched (coding of the ten groups not documented)
Y|74|undocumented_y_74|c|-|UNDOCUMENTED: codebook lists 'urban and rural quintiles' here but values do not match
Y|75|undocumented_y_75|c|-|UNDOCUMENTED: punched 1/5 (codebook lists home ownership here; values do not match)
Y|76|undocumented_y_76|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
Y|77|undocumented_y_77|c|-|UNDOCUMENTED: punched 1/5 in a column the codebook calls blank
Y|78|family_filter|c|FILTER_FAM|Filter: status of family data (use 1, 2, 5 for family estimates)
Y|79-80|weight_family|w|-|Family weight (card Y): 0 for related secondaries whose data sit on the primary's card
Z|8|fam_other_debt_assigned|c|ASSIGNED|FAMILY CARD: was non-real-estate debt assigned
Z|9-11|fam_non_real_estate_debt|n10|-|Family debt other than real estate debt (punched in $10s). ONLY FILLED IN FOR MULTI-SPENDING-UNIT FAMILIES (351 cases); use fam_total_debt for all families
Z|12|fam_non_real_estate_debt_class|c|DEBT8|Family debt other than real estate (class)
Z|13-16|fam_total_debt|n10|-|Family total debt (punched in $10s)
Z|17|fam_total_debt_class|c|DEBT8|Family total debt (class)
"""

# Columns that only repeat another card (verified below, then dropped)
DUPLICATES = [  # (card, first col, card, first col, width, what)
    ("1", 49, "5", 14, 5, "card 1 cols 49-53 = card 5 cols 14-18 (unincorporated business income, profit, loss)"),
    ("1", 54, "7", 18, 5, "card 1 cols 54-58 = card 7 cols 18-22 (business saving)"),
    ("1", 59, "8", 8, 1, "card 1 col 59 = card 8 col 8 (business accounts separate)"),
    ("1", 60, "9", 45, 3, "card 1 cols 60-62 = card 9 cols 45-47 (business money withdrawn, invested, worth)"),
    ("2", 40, "1", 38, 1, "card 2 col 40 = card 1 col 38 (first digit of occupation)"),
    ("2", 41, "1", 47, 1, "card 2 col 41 = card 1 col 47 (1949 income class)"),
    ("1", 47, "5", 63, 1, "card 1 col 47 = card 5 col 63 (1949 income class)"),
    ("7", 55, "6", 55, 20, "card 7 cols 55-74 = card 6 cols 55-74 (durables)"),
    ("8", 73, "6", 73, 2, "card 8 cols 73-74 = card 6 cols 73-74"),
    ("Z", 78, "Y", 78, 1, "card Z col 78 = card Y col 78 (family filter)"),
]

# "Cases in which Amount Exceeded the Field" (last pages of the codebook): true dollar values
OVER_FIELD = {
    "house_value": {"0936": 100000},
    "house_debt_payments_1949": {"0821": 15500},
    "house_principal_payments_1949": {"0821": 15500},
    "consumer_debt_change_amount": {"0947": 10000, "1560": 10000, "1261": 13400},
    "business_saving_amount": {"0726": 16000, "2347": 10000, "0868": 12000, "2669": 10220, "0969": 14715},
    "contractual_saving": {"0821": 16900, "2411": 11500},
    "residual_expenditures": {
        "0507": 15868, "2035": 18325, "0517": 10899, "2151": 12000, "0711": 11170, "2231": 10264,
        "0764": 12469, "2232": 10159, "0822": 11800, "2268": 10084, "0846": 10292, "2363": 14525,
        "0868": 12391, "2489": 10463, "0886": 12741, "2566": 13075, "1051": 10146, "2620": 17042,
        "1284": 10338, "2631": 13166, "1357": 12805, "2647": 15952, "1430": 10280, "3169": 17838,
        "1472": 11203, "3281": 10139, "1491": 10244, "3326": 13488, "1792": 16410, "3382": 10787,
        "1809": 15410, "3481": 14661, "1844": 19686, "3641": 12062, "2023": 12195},
    "liquid_assets": {"0507": 111452, "2567": 122221},
    "stock_net_purchase_amount": {"0698": 10100, "1953": 12100},
    "fam_liquid_assets": {"0507": 111145, "2567": 122220, "3640": 134860},
    "fam_non_real_estate_debt": {"3455": 10500, "3479": 19990},
}

MIDPOINT = {"1": 21, "2": 30, "3": 40, "4": 50, "5": 60, "6": 70}  # age class -> rough midpoint (65+ -> 70)


# --------------------------------------------------------------------------- helpers
# Keypunch errors in the source data.  {variable: {interview: (value as punched, corrected value, evidence)}}
# house_debt: punched ten times too large; the bracket code (card 3 col 62), the debt-to-value class (col 51)
# and total debt (card 9) all agree with one tenth of the punched figure.
# total_debt: punched in dollars instead of $100s; the components add up to one hundredth of the punched
# figure and SRC itself coded family total debt as not ascertained.  Set to missing rather than guessed.
SUSPECT_PUNCHES = {
    "house_debt": {
        "0052": (70000.0, 7000.0, "class $5,000-7,499, 50-59% of a $12,000 house, total debt $7,500"),
        "0330": (80000.0, 8000.0, "class $7,500-9,999, house worth $16,000, total debt $8,800"),
        "2412": (55000.0, 5500.0, "class $5,000-7,499, 50-59% of a $10,000 house, total debt $5,600"),
        "2672": (35400.0, 3540.0, "class $3,000-4,999, house worth $13,000, total debt $3,500"),
    },
    "total_debt": {
        "0299": (250000.0, np.nan, "components sum to $2,500, income $2,690; family total debt coded NA"),
        "1235": (219000.0, np.nan, "components sum to $2,100, income $3,015; family total debt coded NA"),
    },
}


def parse_layout():
    specs = []
    for line in LAYOUT.strip().splitlines():
        card, cols, name, kind, lab, desc = [p.strip() for p in line.split("|", 5)]
        a, _, b = cols.partition("-")
        specs.append(dict(card=card, start=int(a), end=int(b or a), name=name, kind=kind,
                          labels=None if lab == "-" else lab, desc=desc))
    return specs


def punch(s):
    """Normalise zone punches: '-' -> X, '&' -> Y."""
    return s.replace("-", "X").replace("&", "Y")


def parse_amount(raw, mult, trail_y):
    """Return (dollars or NaN, status)."""
    s = punch(raw)
    if s.isdigit():
        return float(int(s) * mult), ""
    if not s.strip():
        return np.nan, "blank"
    if s[0] == "Y" and s[1:].isdigit():
        return np.nan, "over_field"
    if s[0] == "X" and s[1:].isdigit():
        return np.nan, "amount_na"
    if s[-1] == "X" and s[:-1].isdigit():
        return np.nan, "na_whether"
    if s[-1] == "Y" and s[:-1].isdigit():
        if trail_y == "under_50":
            return 25.0, "under_50"
        return np.nan, trail_y
    return np.nan, "invalid_punch"


def load(labeled=False):
    """Read the cleaned file with codes as text and amounts as numbers."""
    name = "scf1950_labeled.csv" if labeled else "scf1950_clean.csv"
    num = {s["name"] for s in parse_layout() if s["kind"][0] in "npw"}
    df = pd.read_csv(os.path.join(OUT, name), dtype=str, keep_default_na=False, na_values=[""])
    for c in df.columns:
        if c in num or c in ("saving", "business_saving", "consumer_debt_change", "liquid_assets_change",
                             "re_saving", "stock_net_purchases", "fam_saving", "car_model_year",
                             "car_year_bought", "age_head_midpoint", "is_farm_schedule", "is_primary_su", "owns_home"):
            df[c] = pd.to_numeric(df[c])
    return df


def main():
    rep = []

    def say(msg=""):
        rep.append(msg)
        print(msg)

    # ------------------------------------------------------------------ 1. read + structure checks
    with open(RAW, encoding="latin-1") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    say("1950 SURVEY OF CONSUMER FINANCES (ICPSR 3612) - CLEANING REPORT")
    say("=" * 70)
    say("\n1. FILE STRUCTURE")
    say(f"   lines read: {len(lines):,}   (expected 38,632 = 3,512 x 11)")
    assert len(lines) % 11 == 0, "line count is not a multiple of 11"
    widths = Counter(len(ln) for ln in lines)
    say(f"   line widths: {dict(widths)}   (expected all 80)")
    assert set(widths) == {80}, "not every line is 80 columns"
    assert all(ln[:2] == "59" for ln in lines), "study number is not 59 on every card"
    n = len(lines) // 11
    cards = {c: [] for c in CARDS}
    ids = []
    for i in range(n):
        block = lines[i * 11:(i + 1) * 11]
        assert "".join(b[2] for b in block) == CARDS, f"cards out of order at case {i}"
        idset = {b[3:7] for b in block}
        assert len(idset) == 1, f"interview number differs across cards at case {i}"
        ids.append(block[0][3:7])
        for c, b in zip(CARDS, block):
            cards[c].append(b)
    assert len(set(ids)) == n, "duplicate interview numbers"
    say(f"   cases: {n:,}; every case has cards 1-9, Y, Z in order with one shared interview number")
    say(f"   interview numbers unique, range {min(ids)}-{max(ids)} "
        f"({sum(i < '3001' for i in ids):,} in the 0001-2999 block, {sum(i >= '3001' for i in ids):,} in the 3001+ multiple-SU block)")
    chars = Counter(ch for ln in lines for ch in ln)
    say("   characters present: " + " ".join(f"{repr(k)}:{v:,}" for k, v in sorted(chars.items())))
    say("   -> only digits, blanks, '-' (X punch), '&' (Y punch) and the card letters Y/Z occur")

    specs = parse_layout()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)), "duplicate variable names in layout"

    # documented columns must be fully punched; undocumented columns must be blank
    say("\n2. LAYOUT CHECK (codebook columns vs. what is actually punched)")
    used = {c: set(range(1, 8)) for c in CARDS}
    for s in specs:
        used[s["card"]].update(range(s["start"], s["end"] + 1))
    dup_cols = {c: set() for c in CARDS}
    for c1, a1, c2, a2, w, what in DUPLICATES:
        same = sum(r1[a1 - 1:a1 - 1 + w] == r2[a2 - 1:a2 - 1 + w] for r1, r2 in zip(cards[c1], cards[c2]))
        say(f"   duplicate check: {what}: identical in {same:,}/{n:,} cases -> dropped")
        dup_cols[c1].update(range(a1, a1 + w))
    dup_cols["1"].discard(47)  # kept as income_1949_class_card1
    for c in "3456789Z":
        dup_cols[c].update({79, 80})
    dup_cols["2"].update({79, 80}); dup_cols["4"].update({79, 80})
    for c in CARDS:
        stray = Counter()
        for r in cards[c]:
            for col in range(1, 81):
                if col not in used[c] and col not in dup_cols[c] and r[col - 1] != " ":
                    stray[col] += 1
        blank_in_field = Counter()
        for s in (s for s in specs if s["card"] == c):
            for r in cards[c]:
                if " " in r[s["start"] - 1:s["end"]]:
                    blank_in_field[s["name"]] += 1
        say(f"   card {c}: non-blank punches outside the layout: {dict(stray) or 'none'}; "
            f"blanks inside documented fields: {dict(blank_in_field) or 'none'}")

    # weights repeated on every card
    w1 = [r[78:80] for r in cards["1"]]
    for c in "2456789":
        diff = [(ids[i], r[78:80], w1[i]) for i, r in enumerate(cards[c]) if r[78:80] != w1[i]]
        if diff:
            say(f"   card {c} weight differs from card 1 in {len(diff)} case(s): {diff} -> card 1 weight used")
    say("   cards 2 and 4-9 otherwise repeat the card 1 weight; cards 3 and Y/Z carry their own")

    # ------------------------------------------------------------------ 3. decode
    data = {"interview_number": ids}
    status_counts = {}
    for s in specs:
        raw = [r[s["start"] - 1:s["end"]] for r in cards[s["card"]]]
        kind = s["kind"]
        if kind == "s":
            data[s["name"]] = [punch(x) for x in raw]
        elif kind == "c":
            data[s["name"]] = [punch(x) for x in raw]
        elif kind == "w":
            data[s["name"]] = [int(x) for x in raw]
        elif kind == "pct":
            vals = [parse_amount(x, 1, "invalid_punch") for x in raw]
            data[s["name"]] = [v for v, _ in vals]
            st = ["over_99_pct" if t == "over_field" else t for _, t in vals]
            if any(st):
                data[s["name"] + "_status"] = st
                status_counts[s["name"]] = Counter(t for t in st if t)
        else:  # n1 / n10 / n100
            k, _, trail = kind.partition(":")
            mult = int(k[1:])
            trail = trail or ("under_50" if mult == 100 else "invalid_punch")
            vals = [parse_amount(x, mult, trail) for x in raw]
            data[s["name"]] = [v for v, _ in vals]
            st = [t for _, t in vals]
            if any(st):
                data[s["name"] + "_status"] = st
                status_counts[s["name"]] = Counter(t for t in st if t)
    df = pd.DataFrame(data)

    say("\n3. DOLLAR FIELDS: special punches set to missing (counts)")
    for k, v in status_counts.items():
        say(f"   {k:34s} " + ", ".join(f"{a}={b}" for a, b in sorted(v.items())))

    # patch over-field cases from the codebook list
    say("\n4. AMOUNTS THAT EXCEEDED THE PUNCH FIELD (restored from the codebook's own list)")
    idx = {v: i for i, v in enumerate(ids)}
    for var, fixes in OVER_FIELD.items():
        ok = 0
        for iv, val in fixes.items():
            i = idx[iv]
            if df.at[i, var + "_status"] == "over_field":
                df.at[i, var] = float(val)
                df.at[i, var + "_status"] = "restored_from_codebook"
                ok += 1
            else:
                say(f"   WARNING {var} case {iv}: codebook lists an over-field value but field is not Y-punched")
        say(f"   {var:34s} {ok}/{len(fixes)} cases restored")
    # a punched zero in these fields means "does not have one", not a price of $0
    say("\n   zero recoded to missing (status 'inapplicable') where 0 means 'does not own / rent / plan to buy':")
    for var in ["house_value", "monthly_rent", "house_plan_price", "car_price", "car_plan_price"]:
        z = df[var] == 0
        df.loc[z, var] = np.nan
        df.loc[z, var + "_status"] = "inapplicable"
        say(f"   {var:34s} {int(z.sum()):,} cases")
    # keypunch errors in the source that three independent fields agree on
    say("\n   keypunch errors repaired or blanked (see SUSPECT_PUNCHES in the script for the evidence):")
    for var, fixes in SUSPECT_PUNCHES.items():
        for iv, (expect, new, why) in fixes.items():
            i = idx[iv]
            if df.at[i, var] != expect:
                say(f"   WARNING {var} case {iv}: expected {expect}, found {df.at[i, var]} - not changed")
                continue
            df.at[i, var] = new
            df.at[i, var + "_status"] = "repaired_keypunch" if new == new else "suspect_keypunch"
            say(f"   {var} case {iv}: {expect:,.0f} -> {'missing' if new != new else format(new, ',.0f')}  ({why})")
    # card Z cols 9-11 were only filled in for families of two or more spending units
    single = df["fam_su_structure"].isin(["0", "1", "2"]) & (df["fam_non_real_estate_debt_status"] == "na_whether")
    df.loc[single, "fam_non_real_estate_debt_status"] = "not_punched_single_su_family"
    say(f"   fam_non_real_estate_debt: '00X' in {int(single.sum()):,} single-spending-unit families means the field was "
        "not filled in for them, not that the debt is unknown")
    left = {v: int((df[v + "_status"] == "over_field").sum()) for v in status_counts
            if (df[v + "_status"] == "over_field").any()}
    say(f"   over-field cases still missing (not in the codebook list): {left or 'none'}")

    # ------------------------------------------------------------------ 5. derived variables
    def signed(amount, sign, pos, neg, zero=("0",)):
        out = pd.Series(np.nan, index=df.index)
        a, sg = df[amount], df[sign]
        out[sg.isin(pos)] = a[sg.isin(pos)]
        out[sg.isin(neg)] = -a[sg.isin(neg)]
        out[sg.isin(zero)] = 0.0
        return out

    df["saving"] = signed("saving_amount", "saving_sign", ["1", "2"], ["3"])
    df["business_saving"] = signed("business_saving_amount", "business_saving_sign", ["1"], ["2"])
    df["consumer_debt_change"] = signed("consumer_debt_change_amount", "consumer_debt_change_dir", ["2"], ["1"])
    df["liquid_assets_change"] = signed("liquid_assets_change_amount", "liquid_assets_change_sign", ["1", "2"], ["3"])
    df["re_saving"] = signed("re_saving_amount", "re_saving_sign", ["1"], ["2"])
    df["stock_net_purchases"] = signed("stock_net_purchase_amount", "stock_trading_1949", ["1", "3"], ["2", "4"], zero=("0", "6"))
    df["fam_saving"] = signed("fam_saving_amount", "fam_saving_sign", ["1", "2"], ["3"])
    df["sample_point"] = df["sample_book_number"].str[:2].map(SAMPLE_POINTS)
    df["car_model_year"] = df["car_model_year_code"].map(YEAR_NUM)
    df["car_year_bought"] = df["car_year_bought_code"].map(YEAR_NUM)
    df["age_head_midpoint"] = df["age_head"].map(MIDPOINT)
    df["is_farm_schedule"] = (df["occupation_group"] == "8").astype(int)  # 410, as the codebook states
    df["is_primary_su"] = df["su_composition"].str[0].isin(["0", "1", "3"]).astype(int)
    df["owns_home"] = df["housing_tenure"].map(lambda v: 1 if v in ("1", "5", "7") else (np.nan if v in ("X",) else 0))
    df.loc[df["weight_housing"] == 0, "owns_home"] = np.nan  # farm + related secondaries: no housing data
    derived = {
        "saving": "Savings A, SIGNED: + saving, - dissaving (saving_amount x saving_sign)",
        "business_saving": "Business saving, signed",
        "consumer_debt_change": "Change in consumer debt, signed: + = debt rose, - = debt fell",
        "liquid_assets_change": "Change in liquid assets since a year ago, signed",
        "re_saving": "Net saving from other real estate transactions, signed",
        "stock_net_purchases": "Net purchases (+) or sales (-) of securities",
        "fam_saving": "Family Savings A, signed",
        "sample_point": "Sample point name (from first two digits of sample book number)",
        "car_model_year": "Car model year as a number (1935 = 1935 or earlier)",
        "car_year_bought": "Year car bought as a number (1935 = 1935 or earlier)",
        "age_head_midpoint": "Rough midpoint of head's age class (65+ set to 70) - for plotting only",
        "is_farm_schedule": "1 = farm operator interviewed on the farm schedule (410 cases; many nonfarm questions not asked). "
                            "Note 67 of them are in nonfarm sampling groups, so sampling_ratio 7/8 does NOT identify them",
        "is_primary_su": "1 = primary spending unit in its dwelling",
        "owns_home": "1 = owns home (tenure 1, 5, 7); missing where housing data are not on this record",
    }
    say("\n5. DERIVED VARIABLES")
    for k, v in derived.items():
        say(f"   {k:24s} {v}")

    # ------------------------------------------------------------------ 6. consistency checks
    say("\n6. INTERNAL CONSISTENCY CHECKS (these validate the column layout)")

    def check(label, ok, base):
        ok, base = int((ok & base).sum()), int(base.sum())
        say(f"   {label:68s} {ok:,}/{base:,} ({100 * ok / max(base, 1):.1f}%)")

    inc, dis, tax = df["income_1949"], df["disposable_income_1949"], df["income_tax_1949"]
    b = inc.notna() & dis.notna() & tax.notna()
    check("disposable income = income - tax", (inc - tax - dis).abs() <= 1, b)
    edges = [0, 1000, 2000, 3000, 4000, 5000, 7500, 10000, np.inf]
    cls = pd.cut(inc, edges, right=False, labels=list("12345678")).astype(object)
    b = inc.notna() & df["income_1949_class"].isin(list("12345678"))
    check("income class (card 5 col 63) agrees with dollar income", cls == df["income_1949_class"], b)
    b = inc.notna() & (inc > 0) & df["tax_pct_income"].notna() & tax.notna()
    check("tax percent agrees with tax / income (within 1 point)",
          ((100 * tax / inc).round() - df["tax_pct_income"]).abs() <= 1, b)
    a, c = pd.to_numeric(df["adults_in_su"], errors="coerce"), pd.to_numeric(df["children_under_18"], errors="coerce")
    p = pd.to_numeric(df["people_in_su"], errors="coerce")
    b = a.notna() & c.notna() & p.notna() & (a < 4) & (c < 7) & (p < 8)
    check("people in SU = adults + children", a + c == p, b)
    b = df["saving"].notna() & inc.notna() & (inc > 0) & df["saving_pct_income"].notna() & (df["saving_pct_income"] < 99)
    check("saving percent agrees with |saving| / income (within 1 point)",
          ((100 * df["saving"].abs() / inc).round() - df["saving_pct_income"]).abs() <= 1, b)
    sedges = [-np.inf, -499.5, -99.5, -0.5, 0.5, 99.5, 199.5, 499.5, 999.5, 1999.5, np.inf]
    scl = pd.cut(df["saving"], sedges, labels=["9", "8", "7", "0", "6", "5", "4", "3", "2", "1"]).astype(object)
    b = df["saving"].notna() & df["saving_class"].isin(list("0123456789"))
    check("saving class (card 7 col 48) agrees with signed saving", scl == df["saving_class"], b)
    parts = df[["af_bonds", "other_govt_bonds", "savings_accounts", "checking_account"]]
    b = parts.notna().all(axis=1) & df["liquid_assets"].notna()
    check("liquid assets = bonds + savings + checking (within $50)",
          (parts.sum(axis=1) - df["liquid_assets"]).abs() <= 50, b)
    b = df["durables_net_spending"].notna() & df["durables_spending_1949"].notna() & (df["car_ownership"] == "0")
    check("no-car units: net durable spending <= gross durable spending",
          df["durables_net_spending"] <= df["durables_spending_1949"], b)
    one = df["family_filter"].eq("1") & df["fam_su_structure"].isin(["0", "1", "2"])
    b = one & inc.notna() & df["fam_income_1949"].notna()
    check("single-SU families: family income = SU income cut to $10 ",
          (inc - df["fam_income_1949"]).between(0, 9.99), b)
    b = df["housing_tenure"].isin(["1"]) & df["house_value"].notna() & df["house_value_class"].isin(list("123456789"))
    hcl = pd.cut(df["house_value"], [0, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000, np.inf],
                 right=False, labels=list("123456789")).astype(object)
    check("house value class (card 3 col 61) agrees with dollar value", hcl == df["house_value_class"], b)

    b = df["fam_saving"].notna() & df["fam_saving_class"].isin(list("0123456789")) & df["family_filter"].isin(["1", "2", "5"])
    fcl = pd.cut(df["fam_saving"], [-np.inf, -495, -95, -5, 5, 95, 195, 495, 995, 1995, np.inf],
                 labels=["9", "8", "7", "0", "6", "5", "4", "3", "2", "1"]).astype(object)
    check("family saving class (card Y col 70) agrees with signed family saving", fcl == df["fam_saving_class"], b)
    ym = df["years_married"].map({"1": "1", "2": "1", "3": "2", "4": "3", "5": "4", "6": "5", "7": "6", "8": "7"})
    b = ym.notna()
    check("family code (card 1 col 63) agrees with years married (col 30)", df["family_composition_code"].str[0] == ym, b)
    b = df["car_price"].notna() & df["car_net_price"].notna() & (df["car_price"] > 0)
    check("car price net of trade-in <= car price", df["car_net_price"] <= df["car_price"], b)
    one8 = df["family_filter"].eq("1") & df["fam_su_structure"].eq("0") & df["fam_durables_net_spending"].notna() & df["durables_net_spending"].notna()
    check("single-SU families: family durables = SU durables cut to $10",
          (df["durables_net_spending"] - df["fam_durables_net_spending"]).between(0, 9.99), one8)
    say("   NOTE card Y col 62 (codebook: family durable spending class) matches the dollar amount only for")
    say("        codes 0, 7, 8, 9; kept as fam_durables_class_unreliable and should not be used.")
    say("   NOTE card 8: the codebook's col 56 (accrual flag) was never punched, so cols 57-71 of the codebook")
    say("        sit one column to the left in the data (56-70); col 71 holds an undocumented change class.")
    say("   NOTE house_debt_class: the misses are NOT a layout problem. 27 debts of $4,000-4,900 carry class 5")
    say("        ($5,000-7,499) while 48 others in the same band carry class 4; the class 4/5 boundary is illegible")
    say("        in the codebook. Use the dollar amount, not the class. The disposable-income misses are")
    say("        arithmetic slips in the source data.")
    hd = df["house_debt"] > df["total_debt"] + 100
    say(f"   NOTE house debt exceeds total debt in {int(hd.sum())} cases (after the four repairs above); in three of them")
    say("        (1475, 1517, 1622) house debt is about twice total debt. Left as punched.")
    say("   bracket code vs. dollar amount, every card (catches shifted columns):")
    H9 = [1, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 30000]
    D8 = [1, 950, 1950, 2950, 3950, 4950, 7450, 9950]
    S6 = [1, 200, 500, 1000, 2000, 5000]
    P7 = [1, 100, 200, 300, 500, 1000, 1500]
    HD = [1, 500, 1000, 3000, 5000, 7500, 10000]
    W9 = [1, 100, 500, 1000, 2000, 3000, 4000, 5000, 10000]
    I8 = [-0.5, 1000, 2000, 3000, 4000, 5000, 7500, 10000]
    L8 = [1, 200, 500, 1000, 2000, 5000, 10000, 25000]
    for cvar, dvar, lo, codes in [
            ("monthly_rent_class", "monthly_rent", [1, 20, 30, 40, 50, 75, 100, 150, 200], "123456789"),
            ("house_debt_class", "house_debt", HD, "1234567"),
            ("house_debt_payments_class", "house_debt_payments_1949", P7, "1234567"),
            ("house_principal_class", "house_principal_payments_1949", P7, "1234567"),
            ("house_plan_price_class", "house_plan_price", H9, "123456789"),
            ("other_re_debt_class", "other_real_estate_debt", HD, "1234567"),
            ("disposable_income_class", "disposable_income_1949", I8, "12345678"),
            ("income_tax_class", "income_tax_1949", [1, 50, 100, 200, 500, 1000, 2000, 5000], "12345678"),
            ("income_unincorp_class", "income_unincorp_business", W9, "123456789"),
            ("income_wages_class", "income_wages", W9, "123456789"),
            ("installment_debt_increase_class", "installment_debt_increase", S6, "123456"),
            ("contractual_saving_class", "contractual_saving", S6, "123456"),
            ("af_bonds_class", "af_bonds", S6, "123456"),
            ("other_bonds_class", "other_govt_bonds", S6, "123456"),
            ("savings_accounts_class", "savings_accounts", S6, "123456"),
            ("checking_class", "checking_account", S6, "123456"),
            ("liquid_assets_class", "liquid_assets", [1, 200, 500, 1000, 2000, 5000, 10000], "1234569"),
            ("life_insurance_premium_class", "life_insurance_premiums", [1, 50, 100, 200, 500, 1000], "123456"),
            ("total_debt_class", "total_debt", D8, "12345678"),
            ("personal_debt_class", "personal_debt", D8, "12345678"),
            ("other_debt_class", "other_debt", D8, "12345678"),
            ("fam_income_class", "fam_income_1949", I8, "12345678"),
            ("fam_liquid_assets_class", "fam_liquid_assets", L8, "12345678"),
            ("fam_non_real_estate_debt_class", "fam_non_real_estate_debt", D8, "12345678"),
            ("fam_total_debt_class", "fam_total_debt", D8, "12345678")]:
        want = pd.cut(df[dvar], lo + [np.inf], right=False, labels=list(codes)).astype(object)
        b = want.notna() & df[cvar].isin(list(codes))
        check(f"   {cvar} vs {dvar}", want == df[cvar], b)

    # ------------------------------------------------------------------ 7. weights
    say("\n7. WEIGHTS")
    say(f"   spending-unit weight (card 1): values {sorted(int(x) for x in df['weight'].unique())}, sum = {df['weight'].sum():,}")
    say(f"   housing weight (card 3): 0 for {int((df['weight_housing'] == 0).sum()):,} units "
        f"(farm schedule or related secondary); otherwise equals the card 1 weight in "
        f"{int(((df['weight_housing'] == df['weight']) & (df['weight_housing'] > 0)).sum()):,} cases")
    say(f"   family weight (card Y): 0 for {int((df['weight_family'] == 0).sum()):,} related secondary units; "
        f"sum = {df['weight_family'].sum():,}")
    say("   -> use `weight` with spending-unit variables, `weight_housing` with card 3 housing variables,")
    say("      `weight_family` with fam_* variables (keep family_filter 1, 2, 5).")
    w = df["weight"]
    v = inc.notna()
    order = np.argsort(inc[v].values)
    cw = np.cumsum(w[v].values[order]) / w[v].sum()
    med = inc[v].values[order][np.searchsorted(cw, 0.5)]
    say(f"   benchmark: weighted mean 1949 SU income = ${np.average(inc[v], weights=w[v]):,.0f}, "
        f"weighted median = ${med:,.0f}; unweighted mean = ${inc[v].mean():,.0f}")
    neg = df["income_1949_status"] == "negative"
    inc0 = inc.where(~neg, 0.0)
    v0 = inc0.notna()
    order = np.argsort(inc0[v0].values, kind="stable")
    cw = np.cumsum(w[v0].values[order]) / w[v0].sum()
    med0 = inc0[v0].values[order][np.searchsorted(cw, 0.5)]
    say(f"   the 32 negative incomes have no amount punched; counting them as $0 instead of missing gives "
        f"mean ${np.average(inc0[v0], weights=w[v0]):,.0f}, median ${med0:,.0f}")
    say("   PUBLISHED (Federal Reserve Bulletin, Aug 1950, '1950 Survey of Consumer Finances, Part III', Table 1):")
    say("      mean $3,270, median $2,700; 3,512 spending units; 706 units (15.8% weighted) with $5,000 or more")
    cls = df["income_1949_class"]
    known = cls.isin(list("123456789"))
    grp = {"under $1,000 (incl. negative)": ["1", "9"], "$1,000-1,999": ["2"], "$2,000-2,999": ["3"],
           "$3,000-3,999": ["4"], "$4,000-4,999": ["5"], "$5,000-7,499": ["6"], "$7,500-9,999": ["7"],
           "$10,000 and over": ["8"]}
    pub = [14, 19, 21, 19, 11, 11, 2, 3]
    say("      income class                      this file   published")
    for (name, codes), pb in zip(grp.items(), pub):
        mine = w[cls.isin(codes)].sum() / w[known].sum() * 100
        say(f"      {name:32s} {mine:8.1f}%   {pb:8d}%")
    top = cls.isin(["6", "7", "8"])
    say(f"      $5,000 or more: {int(top.sum())} cases, {w[top].sum() / w.sum() * 100:.1f}% weighted (published: 706, 15.8%)")
    for name, codes, pm, pn in [("professional", ["1"], 5350, 287), ("managerial + self-employed", ["2", "3"], 5630, 466),
                                ("clerical and sales", ["4"], 3260, 486), ("farm operators", ["8"], 2570, 410)]:
        g = df["occupation_group"].isin(codes)
        gv = g & v0
        say(f"      {name:28s} cases {int(g.sum())} (published {pn}); mean ${np.average(inc0[gv], weights=w[gv]):,.0f} "
            f"(published ${pm:,})")

    # ------------------------------------------------------------------ 8. write
    os.makedirs(OUT, exist_ok=True)
    front = ["interview_number", "weight", "weight_housing", "weight_family", "housing_filter", "family_filter"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]
    df.to_csv(os.path.join(OUT, "scf1950_clean.csv"), index=False)

    lab = df.copy()
    for s in specs:
        if s["kind"] == "c" and s["labels"]:
            m = LABELS[s["labels"]]
            lab[s["name"]] = df[s["name"]].map(lambda v, m=m: m.get(v, v))
    lab.to_csv(os.path.join(OUT, "scf1950_labeled.csv"), index=False)

    unl = Counter()
    with open(os.path.join(OUT, "scf1950_variables.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["variable", "card", "col_start", "col_end", "type", "punched_unit", "description",
                     "n_missing", "value_labels"])
        wr.writerow(["interview_number", "all", 4, 7, "id", "", "Interview number (unique spending-unit id)", 0, ""])
        for s in specs:
            kind = s["kind"]
            typ = {"c": "categorical", "s": "string", "w": "weight", "pct": "percent"}.get(kind, "dollars")
            unit = "$" + kind.split(":")[0][1:] if kind[0] == "n" else ""
            m = LABELS.get(s["labels"], {}) if s["labels"] else {}
            if kind == "c" and m:
                seen = set(df[s["name"]].unique())
                for code in seen - set(m):
                    unl[s["name"]] += int((df[s["name"]] == code).sum())
            nmiss = int(df[s["name"]].isna().sum()) if kind[0] in "np" else ""
            wr.writerow([s["name"], s["card"], s["start"], s["end"], typ, unit, s["desc"], nmiss,
                         "; ".join(f"{k}={v}" for k, v in m.items())])
        for k, v in derived.items():
            wr.writerow([k, "derived", "", "", "derived", "", v, int(df[k].isna().sum()), ""])
        for k in status_counts:
            wr.writerow([k + "_status", "derived", "", "", "status", "",
                         f"Why {k} is missing or adjusted (blank = ordinary punched value)", "", ""])

    say("\n8. CODES PUNCHED BUT NOT LISTED IN THE CODEBOOK LABELS (left as raw codes; cases affected)")
    for k, v in sorted(unl.items(), key=lambda kv: -kv[1]):
        say(f"   {k:34s} {v:,}")
    say("\n9. OUTPUT")
    say(f"   clean/scf1950_clean.csv      {df.shape[0]:,} rows x {df.shape[1]} columns")
    say(f"   clean/scf1950_labeled.csv    same, categorical codes replaced by labels")
    say(f"   clean/scf1950_variables.csv  data dictionary")
    with open(os.path.join(OUT, "cleaning_report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
