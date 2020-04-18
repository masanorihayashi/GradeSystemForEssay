#encoding:utf-8

import glob

path = "/home/lr/hayashi/github/GradeSystemForEssay/cefrj/alignment/A1/*"

test ="""
<result>
<sentence psn="ns">
I like bread and milk but I do n't eat in breakfast rise and JP  .
</sentence>
 <sentence psn="st">
I like breakfast but I do n't eat rice and miso soup for breakfast  .
</sentence>
<trial no="01a">
I like <msf crr="bread">breakfast</msf> <oms>and</oms> <oms>milk</oms> but I do n't eat <oms>in</oms> <oms>breakfast</oms> <msf crr="rise">rice</msf> and <add>miso</add> <add>    soup</add> <add>for</add> <msf crr="JP">breakfast</msf> .
</trial>
</result>

"""

from bs4 import BeautifulSoup as bs

dat = glob.glob(path)


for p in sorted(dat):
    text = ""
    with open(p, "r") as f:
        for i in f:
            text += i
    soup = bs(text, features="html.parser")

    out = soup.find_all(no="01a")

    oms_count = 0
    add_count = 0
    msf_count = 0

    for i in out:
        oms = i.find_all("oms")
        add = i.find_all("add")
        msf = i.find_all("msf")

        oms_count += len(oms)
        add_count += len(add)
        msf_count += len(msf)

    print(p)
    print(oms_count)
    print(add_count)
    print(msf_count)
