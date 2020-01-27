#encoding:utf-8

path = "/home/lr/hayashi/github/GradeSystemForEssay/cefrj/alignment/A1/"

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

soup = bs(test, features="html.parser")

out = soup.find(no="01a")


print(out.find_all("oms"))
a = out.find_all("oms")
print(out.find_all("add"))
print(out.find_all("msf"))

for i in a:
    print(i.string)
