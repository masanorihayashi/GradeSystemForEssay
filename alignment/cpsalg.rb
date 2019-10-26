#encoding: utf-8
require 'kconv'
#
# corpus alignment 
#
#
$version=0.02
# date 2009.09.18-
# 学生の文に " ,"  もしくは "X,X" あるいは" ,X" の形がある場合に，
# ", " に変換してから処理する変更をする．" ."も同様に処理する．
#
# outputをxmlにする．
# 
#
class DeltaClass
    attr_accessor :a, :b
    def initialize
        @a=0
        @b=0
    end
end

class PathClass
    attr_accessor :a, :b
    def initialize
        @a=-1
        @b=-1
    end
end

#--------------------- 2009.09.28 ---
class DistanceScore
    attr_accessor :score, :distance, :umstr, :mcstr
    def initialize
	@score=0.0
	@distance=0.0
	@umstr=0
	@mcstr=0
    end
end

#----------------- 2009.09.25 
class DpResultS
    attr_accessor :score, :len, :a, :b, :c, :isize, :jsize, :drs, :mc, :um, :as, :bs
#    attr_accessor :score, :len, :a, :b, :c, :isize, :jsize, :drs

    def initialize(i)
        @a = Array.new(i)
        @b = Array.new(i)
        @c = Array.new(i)
        @isize = Array.new(i)
        @jsize = Array.new(i)
        @score = 0.0
        @len = 0
        @drs = Array.new(i)
	@mc= Array.new(i) #--- 2009.09.25
	@um= Array.new(i) #--- 2009.09.25
#	@as= Array.new(i) #--- 2009.09.25
#	@bs= Array.new(i) #--- 2009.09.25
    end
end

class Dp 
    attr_accessor :aeqb, :aneb, :aneastr, :bneastr, :aeqastr, :beqastr

    def initialize (a=1.0, b=0.0, c=0.0, d=0.0, e=0.0, f=0.0)
        #/* d(a,b) の 類似度 */
        @aeqb    = a; #/* `a == b'  */
        @aneb    = b; #/* `a !=b' */
        @aneastr = c; #/* `a != *' */
        @bneastr = d; #/* `b != *' */
        @aeqastr = e; #/* `a == *' */
        @beqastr = f; #/* `b == *' */
    end

    # 2つの文字列を受けとり,dpして返すmodeが0のときは文を仮定，
    # 1のときは単語を仮定．文の場合は，再帰的に単語毎にマッチングをする
    # 戻り値は結構複雑だが，DpResultSクラスの入った<result>
    def dpMatchingS (stra, strb, mode)
        result = ""
        if mode==0 then # 文
            stra.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => 'a').encode("UTF-8").gsub(/ +/," ")  # 2以上の空白を1つにする
            #st = stra.split(" ") # st = new StringTokenizer(stra, " ");
            #st = st.encode("UTF-8", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?')
            st = stra.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => 'a').encode("UTF-8").split(" ")
            #ct = st.size         # 配列の要素数 ct = st.countTokens();
            a = Array.new(st.size + 1)  # a = new String[(ct+1)];
            ac1=0
            st.each {|w|
                if w!="" then
                    ac1+=1
                    a[ac1]=w
                end
            }
            #st = strb.split(" ") #st = new StringTokenizer(strb, " ");
            #st = st.encode("UTF-8", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?')
            strb.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => 'a').encode("UTF-8").gsub(/ +/," ")  # 2以上の空白を1つにする
            st = strb.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => 'a').encode("UTF-8").split(" ")
            #ct = st.size #ct = st.countTokens();
            b = Array.new(st.size + 1) #b = new String[(ct+1)];
            ac2=0
            st.each {|w|
                if w!="" then
                    ac2+=1
                    b[ac2]=w
		    #b << w
                end
            }
            if ac2<=0 then # 有効な入力が全くない場合はnull
	    #if b.size<=0 # ??
                return null
            end

        elsif mode==1 then # 単語なので1文字づつ切る．
            ac1 = stra.length # 単語の文字数
            ac2 = strb.length # 単語の文字数
            a = Array.new(ac1+1) #a[0], b[0]は使わない
            b = Array.new(ac2+1) #
            
            for i in 0..(ac1-1) do
                a[(i+1)] = stra.slice(i,1) #=stra.substring(i, i+1);
            end
            for j in 0..(ac2-1) do
                b[(j+1)] = strb.slice(j,1) #=strb.substring(j, j+1);
            end
        else 
            return NULL
        end

        delta = Array.new(ac1+1) # 距離関数dデルタ
        delta.each_index {|i|
            delta[i] = Array.new(ac2+1)
            delta[i].each_index {|j|
                delta[i][j] = DeltaClass.new
            }
        }
        g = Array.new(ac1+1)     # 識別関数g
        g.each_index {|i|
            g[i] = Array.new(ac2+1)
            g[i].each_index {|j|
                g[i][j] = 0.0
            }
        }
        dr = Array.new(ac1+1)    # DPマッチングの結果を入れる？
        dr.each_index {|i|
            dr[i] = Array.new(ac2+1)
        }
#	for i in 0..ac1
#	    for j in 0..ac2
#		dr[i][j]=DpResultS.new(0)
#	    end
#	end

        point = Array.new(ac1+ac2+1) # 得点
        point.each_index {|i|
            point[i] = PathClass.new
        }
        # g = init_g(g, delta, iI, jJ, aeqb, aeqastr, beqastr)
        g[0][0]=0.0
        for i in 1..ac1 do 
            g[i][0] = g[(i-1)][0] + @beqastr
            delta[i][0].a = -1    # @a
            delta[i][0].b = 0     # @b
        end
        for j in 1..ac2 do
            g[0][j] = g[0][(j-1)] + @aeqastr
            delta[0][j].a = 0     # @a
            delta[0][j].b = -1    # @b
        end  #-------------------------------------------------

        for i in 1..ac1 do
            for j in 1..ac2 do
                if mode==0 then  # d(a,b)の計算方法が 文と単語で違う
                    #//ここで再帰的に，単語レベルで呼ぶ
                    dr[i][j] = dpMatchingS(a[i], b[j], 1)
                    distance = dr[i][j].score/dr[i][j].len

#		    print i, ":", j, "  ", a[i], "--", b[j], " score[i][j] = ", dr[i][j].score, "\n"

		    tmpkey=a[i], "--", b[j]
		    ds=DistanceScore.new
		    ds.score=dr[i][j].score
		    ds.distance=distance
		    ds.mcstr=dr[i][j].mc
		    ds.umstr=dr[i][j].um
		    $disscr[tmpkey]=ds

                elsif mode == 1 then     # 単語レベルの場合
                    if a[i] == b[j] then # d(a, b) で a == bの時 charのとき
                        distance = @aeqb
                    else 		 # d(a, b) で a != bの時
                        distance = @aneb
                    end
                end
                g1 = g[(i-1)][j] + @beqastr     # /* g(i-1,j)+d(a,*) */
                g2 = g[(i-1)][(j-1)] + distance	# /* g(i-1,j-1)+d(a,b) */
                g3 = g[i][(j-1)] + @aeqastr	# /* g(i,j-1)+d(*,b) */
                # 同点の時の優先順位 g2>g1>g3
                # g2は斜めなので，最短のパスだから優先
                # g1はa (x) 方向，aが正しいとするので，優先
                if g2>=g1 then
                    if g2>=g3 then # win g2 斜め上に進む
                        g[i][j] = g2
                        delta[i][j].a = -1
                        delta[i][j].b = -1
                    else 	   # win g3 上(y)(b)に進む
                        g[i][j] = g3
                        delta[i][j].a = 0
                        delta[i][j].b = -1
                    end
                elsif g1>=g3 then  # win g1 右(x)(a)に進む
                    g[i][j] = g1
                    delta[i][j].a = -1
                    delta[i][j].b = 0
                else               # win g3 上(y)(b)に進む
                    g[i][j] = g3
                    delta[i][j].a = 0
                    delta[i][j].b = -1
                end
            end
        end
        kk = 0
        point[kk].a = ac1
        point[kk].b = ac2
        while (point[kk].a!=0)||(point[kk].b!=0) do # point[kk]=(0, 0) の間
            x = point[kk].a
            y = point[kk].b
            kk+=1
            point[kk].a = x + delta[x][y].a
            point[kk].b = y + delta[x][y].b
        end
        true_point = Array.new(kk+1)      #true_point=new PathClass[(K+1)];
        for i in 0..kk do
            true_point[i] = PathClass.new #true_point[i]= new PathClass();
        end
        result = DpResultS.new(kk+1)      #result = new DpResultS(K+1);
        result.score = g[ac1][ac2]        # g[I][J]
        result.drs = dr 
        #
        if ac1<ac2 then
            result.len = ac2
        else
            result.len = ac1
        end
        for i in 0..kk do
            true_point[i].a=point[kk-i].a
            true_point[i].b=point[kk-i].b
        end
        prea=1
        preb=1

	e_ct=0 #-a------------------
	l_ct=0 #-b------------------
	m_ct=0 #-match--------------
	u_ct=0 #-unmatch------------
#	print stra, ":", strb, " "
        for i in 1..kk do
            result.isize[i]=prea
            result.jsize[i]=preb
            if true_point[i].a==(true_point[(i-1)].a + 1) then
                result.a[i]=a[prea]
                prea+=1
            else 
                result.a[i]="*"
            end
            
            if true_point[i].b==(true_point[(i-1)].b + 1) then
                result.b[i]=b[preb]
                preb+=1
            else 
                result.b[i]="*"
            end
            #// aを正解と仮定し，b側を評価する
            if    result.a[i]=="*" then         
		result.c[i]="exceeded"
		e_ct+=1
            elsif result.b[i]=="*" then         
		result.c[i]="lacked"
		l_ct+=1
            elsif result.a[i]==result.b[i] then 
		result.c[i]="matched"
		m_ct+=1   #-------------------------------------
            else                                
		result.c[i]="unmatched"
		u_ct+=1 #---------------------------------------
            end
        end
#	print " " + "+" + a_astr_ct.to_s + " -" + b_astr_ct.to_s +
#	    " m" + match_ct.to_s + " u" + unmatch_ct.to_s + "\n"

	result.mc=m_ct
#	result.um=u_ct
	result.um=u_ct+l_ct+e_ct

	termpair = stra + ":" + strb
	$lack[termpair]=l_ct
	$exce[termpair]=e_ct
	$matc[termpair]=m_ct
	$unma[termpair]=u_ct

        return result # result = DpResultS.new(kk+1)
    end
end

class EvaRes 
    attr_accessor :str, :notice, :color, :scored, :score, :unit

    def initialize (i)
        @str = Array.new(i)    #new String[i];
        @str2= Array.new(i)    #-------------------------正解保持---
        @color = Array.new(i)  #new Color[i];
        @notice = Array.new(i) #new String[i]; // add.
        @scored = 0.0
        @score = 0
        @unit = 0
    end
end

class EvaRes2
    attr_accessor :str, :notice, :color, :scored, :score, :unit, :crr, :jud, :distance, :erCount

    def initialize (i)
        @str = Array.new(i)    #new String[i];
        @str2= Array.new(i)    #-------------------------正解保持---
        @color = Array.new(i)  #new Color[i];
        @notice = Array.new(i) #new String[i]; // add.
        @scored = 0.0
        @score = Array.new(i)
	@distance= Array.new(i)
        @unit = 0
	@jud = Array.new(i)
	@crr = Array.new(i) 	#---------途中
	#@erCount=0
    end
end

class  DpCalc2 
    def showResults(drs, str1, str2)
        erCount = 0
        prestate = ""
        size = drs.len + str1.length + 1 #int size=drs.length+str1.length()+1;
        numOfMatched = 0 # i一致する単語数;
        evaRes = EvaRes.new(size)

        ct=0; drs.a.each {|sa| print ct, ":", sa, " "; ct+=1}; print "\n"
        ct=0; drs.b.each {|sb| print ct, ":", sb, " "; ct+=1}; print "\n"
        ct=0; drs.c.each {|sc| print ct, ":", sc, " "; ct+=1}; print "\n"

        for i in 1..(drs.c.length - 1) do
            if drs.c[i]=="" then 
                next 
            end
            if erCount!=0 then evs= " " else evs="" end

            evaRes.notice[erCount] = ""
            if drs.c[i]    == "lacked"  then
                evaRes.str[erCount]=evs +"<oms>"+ drs.a[i] +"</oms> "
                erCount+=1
            elsif drs.c[i] == "exceeded"  then # 余分 「単語単位」
                evaRes.str[erCount] =evs +"<add>"+ drs.b[i] + "</add> "
                erCount +=1
            elsif drs.c[i] == "matched" then
                evaRes.str[erCount] = evs + drs.b[i]
                erCount +=1
                numOfMatched +=1
            else #---------------------部分一致もしくは完全不一致 「文字単位」
                evaRes.str[erCount]= evs
                swt = 0
                jj=drs.drs[drs.isize[i]][drs.jsize[i]].b.length
                for j in 1..jj do
                    nowstate = drs.drs[drs.isize[i]][drs.jsize[i]].c[j]
                    if prestate == nowstate || j==1 then
                        evaRes.str[erCount] = evaRes.str[erCount] + 
                            drs.drs[drs.isize[i]][drs.jsize[i]].b[j]
                    else               # 一個前の文字列が部分一致する場合は黒
                        if  prestate == "matched" then
                            swt = 1
                        else 
                            evaRes.str[erCount]= "[" +
                                evaRes.str[erCount] + "]";
                            evaRes.str[erCount]= 
                                evaRes.str[erCount].gsub("[ "," [")
                            swt = 2
                        end
                        erCount+=1
                        evaRes.str[erCount] =
                            drs.drs[drs.isize[i]][drs.jsize[i]].b[j]
                        evaRes.notice[erCount] =""
                    end
                    prestate=drs.drs[drs.isize[i]][drs.jsize[i]].c[j] # 文字単位
                end
                evaRes.notice[erCount-1]=":msf"

                if prestate == "matched" then
                    evaRes.color[erCount] = "black"
                else
                    evaRes.color[erCount] = "red"
                end
            end
        end
        # スコア計算
        evaRes.unit=erCount
        evaRes.scored=(drs.score)/(drs.len)
        #evaRes.score=(int)((evaRes.scored*100)+0.5)
        evaRes.score=(evaRes.scored*100)

        for i in 0..(evaRes.unit - 1) do
            #print "i=", i, " ", evaRes.str[i], "\n"
            print evaRes.str[i]
            #print evaRes.notice[i], "\n"
            print evaRes.notice[i]
        end
        print "\n"
        print Kconv.tosjis("一致度 ")
        print Kconv.tosjis("単語レベル："), numOfMatched, "/", drs.len,  " "
        #printf "文字レベル：%.2f", drs.score);
        print Kconv.tosjis("文字レベル："), drs.score
        print "/",  drs.len,  "(", evaRes.score, "%)\n"
        print "\n"
        str1 = "";
        str2 = "";
        erCount=0;
    end
    #------------
    def DpCalc2.showResults2(drs, str1, str2)
        erCount = 0
        prestate = ""
        size = drs.len + str1.length + 1 #int size=drs.length+str1.length()+1;
        numOfMatched = 0 # i一致する単語数;
        evaRes = EvaRes.new(size)

#	print "TEST:str2 ", str2, "\n"

        #ct=0; drs.a.each {|sa| print ct, ":", sa, " "; ct+=1}; print "\n"
        #ct=0; drs.b.each {|sb| print ct, ":", sb, " "; ct+=1}; print "\n"
        #ct=0; drs.c.each {|sc| print ct, ":", sc, " "; ct+=1}; print "\n"

        for i in 1..(drs.c.length - 1) do
            if drs.c[i]=="" then 
                next 
            end
            if erCount!=0 then evs=" " else evs="" end

            evaRes.notice[erCount] = ""
            if drs.c[i]    == "lacked"  then
                evaRes.str[erCount]=evs +"<oms>"+ drs.a[i] +"</oms> "
#                evaRes.str[erCount]=evs +"<jud attr=\"oms\">"+ drs.a[i] +"</jud> "
                erCount+=1
            elsif drs.c[i] == "exceeded"  then # 余分 「単語単位」
                evaRes.str[erCount] =evs +"<add>"+ drs.b[i] + "</add> "
#                evaRes.str[erCount]=evs +"<jud attr=\"add\">"+ drs.b[i] +"</jud> "
                erCount +=1
            elsif drs.c[i] == "matched" then
                evaRes.str[erCount] = evs + drs.b[i]
#                evaRes.str[erCount] = evs + "<mat>" + drs.b[i] + "</mat>"
#                evaRes.str[erCount]=evs +"<jud attr=\"mat\">"+ drs.b[i] +"</jud>"
                erCount +=1
                numOfMatched +=1
            else #---------------------部分一致もしくは完全不一致 「文字単位」
                #evaRes.str[erCount]= evs
		evaRes.str[erCount]= evs + "<msf crr=\"" + drs.a[i] + "\">" +
		    drs.b[i] + "</msf>"

		termpair=drs.a[i] + ":" + drs.b[i]

#		evaRes.str[erCount]=evs +"<jud attr=\"msf\" crr=\""+ 
#		    drs.a[i] +"\">"+ drs.b[i] +"</jud>"

#		evaRes.str[erCount]=evs +"<jud attr=\"msf\" crr=\""+ 
#		    drs.a[i] + 
#		    "\" lc=\"" + $lack[termpair].to_s + 
#		    "\" ex=\"" + $exce[termpair].to_s +
#		    "\" m=\""  + $matc[termpair].to_s + 
#		    "\" u=\""  + $unma[termpair].to_s + "\">" + drs.b[i] +"</jud>"

#		evaRes.str[erCount]=evaRes.str[erCount] +
#		    "<TEST: match=" + drs.match.to_s + " unmatch=" + drs.unmatch.to_s + ">"

		erCount+=1
#		print "TEST: match=", drs.match, " unmatch=", drs.unmatch, "\n"
            end

#	    tmpkey=drs.a[i], "--", drs.b[i]
#	    if $disscr[tmpkey] != NIL
#		print tmpkey
#		print " score=", $disscr[tmpkey].score
#		print " distance=", $disscr[tmpkey].distance, "\n"
#	    end
        end
        # スコア計算
        evaRes.unit=erCount
        evaRes.scored=(drs.score)/(drs.len)
        #evaRes.score=(int)((evaRes.scored*100)+0.5)
        evaRes.score=(evaRes.scored*100)

#	print "erCount=", erCount, " "
#	print "drs.score=", drs.score, " "
#	print "drs.len=", drs.len, "\n"

        for i in 0..(evaRes.unit - 1) do
	    printf "%s", evaRes.str[i]
        end
        print "\n"
        str1 = ""
        str2 = ""
        erCount=0
    end

    #------------
    def DpCalc2.restoreResults(drs, str1, str2)
        erCount = 0
        prestate = ""
        size = drs.len + str1.length + 1 #int size=drs.length+str1.length()+1;
        numOfMatched = 0 # i一致する単語数;
        evaRes2 = EvaRes2.new(size)

        for i in 1..(drs.c.length - 1) do
            if drs.c[i]=="" then 
                next 
            end
            if erCount!=0 then evs=" " else evs="" end

	    evaRes2.crr[erCount]=""
	    evaRes2.score[erCount]=0.0
	    evaRes2.distance[erCount]=0.0
            evaRes2.notice[erCount] = ""

            if drs.c[i]    == "lacked"  then
#                evaRes.str[erCount]=evs +"<oms>"+ drs.a[i] +"</oms> "
#                evaRes.str[erCount]=evs +"<jud attr=\"oms\">"+ drs.a[i] +"</jud> "
		evaRes2.str[erCount]=drs.a[i]
		evaRes2.jud[erCount]="oms"
                erCount+=1
            elsif drs.c[i] == "exceeded"  then # 余分 「単語単位」
#                evaRes.str[erCount] =evs +"<add>"+ drs.b[i] + "</add> "
#                evaRes.str[erCount]=evs +"<jud attr=\"add\">"+ drs.b[i] +"</jud> "
		evaRes2.str[erCount]=drs.b[i]
		evaRes2.jud[erCount]="add"
                erCount +=1
            elsif drs.c[i] == "matched" then
#                evaRes.str[erCount] = evs + "<mat>" + drs.b[i] + "</mat>"
#                evaRes.str[erCount]=evs +"<jud attr=\"mat\">"+ drs.b[i] +"</jud>"
		evaRes2.str[erCount]=drs.b[i]
		evaRes2.jud[erCount]="mat"
		evaRes2.score[erCount]=1.0
                erCount +=1
                numOfMatched +=1
            else #---------------------部分一致もしくは完全不一致 「文字単位」
                #evaRes.str[erCount]= evs
#		evaRes.str[erCount]= evs + "<msf crr=\"" + drs.a[i] + "\">" +
#		    drs.b[i] + "</msf>"
		termpair=drs.a[i] + ":" + drs.b[i]
#		evaRes.str[erCount]=evs +"<jud attr=\"msf\" crr=\""+ 
#		    drs.a[i] +"\">"+ drs.b[i] +"</jud>"

		evaRes2.str[erCount]=drs.b[i]
		evaRes2.jud[erCount]="msf"
		evaRes2.crr[erCount]=drs.a[i]
		if $disscr[termpair] != NIL
		    evaRes2.score[erCount]   =$disscr[termpair].score
		    evaRes2.distance[erCount]=$disscr[termpair].distance
		end
		erCount+=1

            end

#	    tmpkey=drs.a[i], "--", drs.b[i]
#	    if $disscr[tmpkey] != NIL
#		print tmpkey
#		print " score=", $disscr[tmpkey].score
#		print " distance=", $disscr[tmpkey].distance, "\n"
#	    end
        end
        # スコア計算
        evaRes2.unit=erCount
        evaRes2.scored=(drs.score)/(drs.len)
        #evaRes2.score=(int)((evaRes.scored*100)+0.5)
        evaRes2.score=(evaRes2.scored*100)

#        for i in 0..(evaRes2.unit - 1) do
#	    printf "%s", evaRes2.str[i]
#        end
#        print "\n"
        #str1 = ""
        #str2 = ""
        #erCount=0

	return evaRes2
    end

    def DpCalc2.showResultsX1(er, str1, str2, strx)

	print "<trial no=\"01", strx, "\">\n"
	for i in 0..er.unit-1 do
	    if er.jud[i]!="mat" then
		print "<", er.jud[i]
		if er.jud[i]=="msf" then
		    print " crr=\"", er.crr[i], "\""
		end
		print ">", er.str[i]
		print "</", er.jud[i], "> "
	    else
		print er.str[i], " "
	    end
	end
	print "\n"
	print "</trial>\n"

#-	print "<trial no=\"02\">\n"
	otstr=""
	flag00=0
	for i in 0..er.unit-1 do
	    # level 1 type 1 -- iとj および jとiが一致することが条件 ---------
	    if er.jud[i]=="msf" then
		flag=0
		max_distance=0.000
		max_j=-1
		for j in i+1..er.unit-1 do
		    if er.jud[j]=="msf" then 
			#--------- 完全一致 ---------------------------------
			if er.crr[i]==er.str[j] && er.str[i]==er.crr[j] then
			    flag=1
#			    print "MATCH!\n"
			    # trans
			    er.jud[i]="trs"
			    #er.crr[i]=(j+1).to_s #, " pos"
			    er.jud[j]="trs"
			    #er.crr[j]=(i+1).to_s #, " pos"
			    j=er.unit # 比較は終了
			    flag00=1
			else
			    tmpkey1=er.str[j], "--", er.crr[i]
			    tmpkey2=er.crr[j], "--", er.str[i]
#			    print tmpkey1, "\n", tmpkey2, "\n"

			    if $disscr[tmpkey1]!=NIL && $disscr[tmpkey2]!=NIL
				then

#				print tmpkey1, " ", $disscr[tmpkey1].distance,
#				" ", tmpkey2, " ", $disscr[tmpkey2].distance, "\n"

				tmpdistance=$disscr[tmpkey1].distance+
				    $disscr[tmpkey2].distance
				if tmpdistance > max_distance then
				    max_distance=tmpdistance
				    max_j=j
				    flag=2
				end
			    end
			end
		    end
		end
		#--------- 完全一致でない場合は，類似度の高いペアを評価--------
		if flag==2 then
		    if max_distance > 1.8 # 適正値は1.8くらいか？
			tmpkey=er.str[i], "--", er.crr[i]
			if $disscr[tmpkey]==NIL then
			    tmpkey=er.crr[i], "--", er.str[i]
			    if $disscr[tmpkey]==NIL then
				print "***\n"
			    end
			end
			#print tmpkey2
#			print "max_distance=", max_distance,
#			"tmpkey_distance=", $disscr[tmpkey].distance, "\n"

			if max_distance > $disscr[tmpkey].distance
			    er.jud[i]="trs"
			    #er.crr[i]=er.str[max_j]
			    tmpcrri=er.crr[i]
			    er.crr[i]=er.crr[max_j]
			    er.jud[max_j]="trs"
			    #er.crr[max_j]=er.str[i] #, " pos"
			    er.crr[max_j]=tmpcrri #, " pos"
			    flag00=1
			end
		    end
		end
#-----------Level 01 - 02  ----------------------------------------------------
	    elsif er.jud[i]=="oms" then # type 2
		flag=0
		max_distance=0.000
		max_j=-1
		for j in i+1..er.unit-1 do
		    if er.jud[j]=="add" then
#			print "MATCHING: t2 ", er.str[i], " VS ", er.str[j], "\n"
			#-----完全一致----------------------------------------
			if er.str[i]==er.str[j] then
#			    print "MATCH!\n"
			    # trans
			    er.jud[i]="trs_oms"
			    #er.crr[i]=(j+1).to_s 
			    er.crr[i]=er.str[j]
			    er.jud[j]="trs_add"
			    #er.crr[j]=(i+1).to_s
			    er.crr[j]=er.str[i]

			    j=er.unit # 比較は終了
			    flag00=1
			else
			    tmpkey1=er.str[i], "--", er.str[j]
			    tmpkey2=er.str[j], "--", er.str[i]
			    tmpdistance=-1
			    if $disscr[tmpkey1]!=NIL then
				tmpdistance=$disscr[tmpkey1].distance
				tmpumstr=$disscr[tmpkey1].umstr
			    elsif $disscr[tmpkey2]!=NIL then
				tmpdistance=$disscr[tmpkey2].distance
				tmpumstr=$disscr[tmpkey2].umstr
			    end
			    if tmpdistance > max_distance then
				max_distance=tmpdistance
				max_umstr = tmpumstr
				max_j=j
				flag=2
			    end
			end
		    end
		end
		if flag==2 then
		    #print "DISTANCE:", max_distance, "\n"
		    if max_distance > 0.85 || (max_distance>=0.75 && max_umstr <=2) 
			# trans
			er.jud[i]="trs_add"
			#er.crr[i]=(j+1).to_s 
#			er.crr[i]=er.str[j]
			er.crr[i]=er.str[max_j]
#			er.jud[j]="trs_oms"
			er.jud[max_j]="trs_oms"
			#er.crr[j]=(i+1).to_s
#			er.crr[j]=er.str[i]
			er.crr[max_j]=er.str[i]
			flag00=1
		    end
		end
#-----------Level 01 - 03  ----------------------------------------------------
	    elsif er.jud[i]=="add" then # type 3
		flag=0
		max_distance=0.000
		max_j=-1
		for j in i+1..er.unit-1 do
#--		    print "word", i, "=", er.str[i], " ", j, "=", er.str[j], " jud j=", er.jud[j], "\n"
		    if er.jud[j]=="oms" then
#--			print "MATCHING: t3 ", er.str[i], " VS ", er.str[j], "\n"
			#-----完全一致----------------------------------------
			if er.str[i]==er.str[j] then
#--			    print "MATCH!\n"
			    # trans
			    er.jud[i]="trs_add"
			    #er.crr[i]=(j+1).to_s 
			    er.crr[i]=er.str[j]
			    er.jud[j]="trs_oms"
			    #er.crr[j]=(i+1).to_s
			    er.crr[j]=er.str[i]

			    j=er.unit # 比較は終了
			    flag00=1
			else
			    tmpkey1=er.str[i], "--", er.str[j]
			    tmpkey2=er.str[j], "--", er.str[i]
			    tmpdistance=-1
			    if $disscr[tmpkey1]!=NIL then
				tmpdistance=$disscr[tmpkey1].distance
				tmpumstr=$disscr[tmpkey1].umstr
			    elsif $disscr[tmpkey2]!=NIL then
				tmpdistance=$disscr[tmpkey2].distance
				tmpumstr=$disscr[tmpkey2].umstr
			    end
#--			    print "(", max_distance, ") ", tmpdistance, 
#--			    " umstr=", tmpumstr, "\n"

			    if tmpdistance > max_distance then
				max_distance=tmpdistance
				max_umstr = tmpumstr
				max_j=j
				flag=2
			    end
			end
		    end
		end
		if flag==2 then
		    if max_distance > 0.85 || (max_distance>=0.75 && max_umstr <=2) # 

#--			print "TRS_ADD i ", er.str[i], " ->j ", er.str[max_j], "\n"

			# trans
			er.jud[i]="trs_add"
			#er.crr[i]=(j+1).to_s 
#			er.crr[i]=er.str[j]
			er.crr[i]=er.str[max_j]
#			er.jud[j]="trs_oms"
			er.jud[max_j]="trs_oms"
			#er.crr[j]=(i+1).to_s
#			er.crr[j]=er.str[i]
			er.crr[max_j]=er.str[i]
			flag00=1
		    end
		end
	    end

	    # 結果の出力のストア
#	    if er.jud[i]!="mat" then
#		print "<", er.jud[i]
#		if er.jud[i]=="msf"|| /^trs/ =~ er.jud[i] then
#		    print " crr=\"", er.crr[i], "\""
#		end
#		print ">", er.str[i]
#		print "</", er.jud[i], "> "
#	    else
#		print er.str[i], " "
#	    end

	    if er.jud[i]!="mat" then
		otstr=otstr + "<" + er.jud[i]
#		if er.jud[i]=="msf"|| /^trs / =~ er.jud[i] then
		if er.jud[i]=="msf"|| /^trs_/ =~ er.jud[i] then
#		    if er.crr[i]!=NIL then
			otstr=otstr + " crr=\"" 
			otstr=otstr + er.crr[i] + "\""
#		    end

		end
		otstr=otstr + ">" + er.str[i]
		otstr=otstr + "</" + er.jud[i] + "> "
	    else
		otstr=otstr + er.str[i] + " "
	    end
	end
#-	print "\n</trial>\n"
	if flag00==1 then
	    print "<trial no=\"02\">\n"
	    print otstr, "\n"
	    print "</trial>\n"
#	    print "\n"
	end

    end

end

$lack = Hash.new
$exce = Hash.new
$matc = Hash.new
$unma = Hash.new

$disscr = Hash.new

#print "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
#print "<text>\n"

state=0
ct=0
blank=0
pre=0
while str1=STDIN.gets do
#    str1=STDIN.gets
    str1.chomp!

    if /^<text id/ =~ str1.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?').encode("UTF-8").gsub!(" ,", ", ") || /<\/text/ =~ str1.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?').encode("UTF-8").gsub!(" ,", ", ") then
	if str1=="<\/text>" then
	    pre=0
	end
	print str1, "\n"
	blank=0
	next
    end
    if str1 == "" then
	if blank==1 then 
	    break
	end
	blank=1
	print "\n"
	next
    else
	blank=0
	str2=STDIN.gets
	if str2!=NIL
	    str2.chomp!
	else
	    print str1, "\n"
	    next
	end
    end

    if  str2=="" then
	print str1, "\n"
	print str2, "\n"
	next
    end
    
#    if str1 == "" then break end
#    print str1, "\n"
#-    if str2 == "" then break end
#    print str2, "\n"

    #--------------------------- ver. 0.002
    tmp=str2.dup
    str2.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?').encode("UTF-8").gsub!(" ,", ", ")
    str2.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?').encode("UTF-8").gsub!(" .", ".")
    str2.encode("UTF-16BE", "UTF-8", :invalid => :replace, :undef => :replace, :replace => '?').encode("UTF-8").gsub!("  ", " ")
    #--------------------------------------

    dp = Dp.new
    #dps = DpResultS.new
    drs = dp.dpMatchingS(str1, str2, 0)
    if drs=="" then
	break
    end
    #
    #test = DpCalc2.new
    # test.showResults(drs, str1, str2)
    #test.showResults2(drs, str1, str2)

#    print "<sentences>\n" #-----------
    if pre==100 then
	print "\n"
	pre=0
    end
    print "<result>\n" 
#    print "<correct>\n"
    print "<sentence psn=\"ns\">\n" #=-
    print str1, "\n"
    print "</sentence>\n"
#-    print "</correct>\n"

    #--------------------------- ver. 0.002

    strx="a"
    if tmp!=str2 then
#-	print "<original_original>\n"
    print "<sentence psn=\"st_\">\n" #=-
#	print "*00:", tmp, "\n"
	print tmp, "\n"
	print "</sentence>\n" #--
	print "<sentence psn=\"st\">\n" #--
	print str2, "\n"
	print "</sentence>\n" #--
#-	print "</original_original>\n"
#-	print "<result>\n"
	print "<trial no=\"00\">\n"
	drs2 = dp.dpMatchingS(str1, tmp, 0)	
	DpCalc2.showResults2(drs2, str1, tmp)
	print "</trial>\n"
#-	print "</result>\n"
	strx="b"
    else
	print "<sentence psn=\"st\">\n" #--
	print str2, "\n"
	print "</sentence>\n" #--
    end
    #--------------------------------------
#-    print "<original>\n"
#    print str2, "\n"
#-    print "</original>\n"
#-    print "</sentences>\n"


#-    print "<result>\n"
#    DpCalc2.showResults2(drs, str1, str2)
    evaResult = DpCalc2.restoreResults(drs, str1, str2)
    DpCalc2.showResultsX1(evaResult, str1, str2, strx)

#-    print "</result>\n"
    # drs.showResults(str1, str2)
    print "</result>\n"

    pre=100
    ct+=1

end
#-print "</text>\n"
