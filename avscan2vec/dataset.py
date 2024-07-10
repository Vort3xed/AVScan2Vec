import os
import sys
import copy
import json
import mmap
import torch
import pickle
import random
import numpy as np
from bpe import Encoder
from datetime import datetime as dt
from torch.utils.data import Dataset

from globalvars import *
from utils import tokenize_label, read_supported_avs


class AVScanDataset(Dataset):

    def __init__(self, data_dir, max_tokens=7, max_chars=20, max_vocab=10000000):
        """Base dataset class for AVScan2Vec.

        Arguments:
        data_dir -- Path to dataset directory
        max_tokens -- Maximum number of tokens per label
        max_chars -- Maximum number of chars per token
        max_vocab -- Maximum number of tokens to track (for masked token / label prediction)
        """

        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.max_vocab = max_vocab

        # Read supported AVs
        av_path = os.path.join(data_dir, "avs.txt")
        self.supported_avs = read_supported_avs(av_path)
        self.avs = sorted(list(self.supported_avs))
        self.av_vocab_rev = [NO_AV] + self.avs
        self.num_avs = len(self.avs)

        # Map each AV to a unique index
        self.av_vocab = {av: idx for idx, av in enumerate(self.av_vocab_rev)}

        # Construct character alphabet
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.alphabet = [char for char in self.alphabet]
        self.SOS_toks = ["<SOS_{}>".format(av) for av in self.avs]
        self.special_tokens = [PAD, CLS, EOS, ABS, BEN] + self.SOS_toks + [SOW, EOW, MASK, UNK]
        self.special_tokens_set = set(self.special_tokens)
        self.alphabet = self.special_tokens + self.alphabet
        self.alphabet_rev = {char: i for i, char in enumerate(self.alphabet)}

        # Load token vocabulary
        vocab_path = os.path.join(data_dir, "vocab.txt")
        self.token_vocab_rev = []
        with open(vocab_path, "r") as f:
            for line in f:
                if len(self.token_vocab_rev) >= self.max_vocab:
                    break
                tok = line.strip()
                self.token_vocab_rev.append(tok)

        # Map each token to a unique index
        self.token_vocab = {tok: idx for idx, tok in enumerate(self.token_vocab_rev)}

        # Zipf distribution for sampling tokens
        self.vocab_size = len(self.token_vocab_rev)
        self.zipf_vals = np.arange(5, self.vocab_size)
        self.zipf_p = 1.0 / np.power(self.zipf_vals, 2.0)
        self.zipf_p /= np.sum(self.zipf_p)

        # Load line offsets
        line_path = os.path.join(data_dir, "line_offsets.pkl")
        with open(line_path, "rb") as f:
            self.line_offsets = pickle.load(f)
        self.line_paths = sorted(list(self.line_offsets.keys()))

        # Get total number of scan reports
        self.num_reports = sum([len(v) for v in self.line_offsets.values()])

        self.test_corpus = '''
<PAD> <UNKNOWN> <EOS> <ABSTAIN> <BENIGN> <SOS_acronis> <SOS_adaware> <SOS_agnitum> <SOS_ahnlabv3> <SOS_alibaba> <SOS_alyac> <SOS_antivir> <SOS_antiyavl> <SOS_apex> <SOS_arcabit> <SOS_avast> <SOS_avastmobile> <SOS_avg> <SOS_avira> <SOS_avware> <SOS_babable> <SOS_baidu> <SOS_baiduinternational> <SOS_bitdefender> <SOS_bitdefendertheta> <SOS_bkav> <SOS_bytehero> <SOS_catquickheal> <SOS_clamav> <SOS_cmc> <SOS_commtouch> <SOS_comodo> <SOS_crowdstrike> <SOS_cybereason> <SOS_cylance> <SOS_cynet> <SOS_cyren> <SOS_drweb> <SOS_egambit> <SOS_elastic> <SOS_emsisoft> <SOS_endgame> <SOS_esafe> <SOS_esetnod32> <SOS_fireeye> <SOS_fortinet> <SOS_fprot> <SOS_fsecure> <SOS_gdata> <SOS_gridinsoft> <SOS_ikarus> <SOS_invincea> <SOS_jiangmin> <SOS_k7antivirus> <SOS_k7gw> <SOS_kaspersky> <SOS_kingsoft> <SOS_lionic> <SOS_malwarebytes> <SOS_max> <SOS_maxsecure> <SOS_mcafee> <SOS_mcafeegwedition> <SOS_microsoft> <SOS_microworldescan> <SOS_nanoantivirus> <SOS_norman> <SOS_nprotect> <SOS_paloalto> <SOS_panda> <SOS_pctools> <SOS_qihoo360> <SOS_rising> <SOS_sangfor> <SOS_sentinelone> <SOS_sophos> <SOS_superantispyware> <SOS_symantec> <SOS_tachyon> <SOS_tencent> <SOS_thehacker> <SOS_totaldefense> <SOS_trapmine> <SOS_trendmicro> <SOS_trendmicrohousecall> <SOS_vba32> <SOS_vipre> <SOS_virobot> <SOS_virusbuster> <SOS_webroot> <SOS_yandex> <SOS_zillya> <SOS_zonealarm> <SOS_zoner> win32 trojan gen w32 malicious generic a agent virus malware variant worm confidence adware b virlock heur score 1 ai tr eldorado 100 troj trj d unsafe high win downloader backdoor behaveslike ml heuristic static generickd application c dropper pe pua engine s of virut v bt vb coinminer pup classic sivis suspicious 0 crypt dinwod ransom 8 kryptik razy genetic 5 ramnit polyransom riskware nabucur allaple zusy e not upatre vtflooder sality wabot genpack bdmj packed 2 o msil trojware delf vflooder adload injector g flooder mg bundler adposhel trojandropper unwanted miner f highconfidence dealply spy autoit k mal installmonster lamer ludbaruma w 4 adw xpack attribute i trojandownloader j famvt downloadguide artemis moderate nimnul symmi virransom regrun smp zbot n rdm h 3 nsis nimda uvpm deepscan 99 small potentially tspy ce autorun cryptor sma cloud banker xfc bscope gen7 ulise installcore b8d96178 virtob eydrop cosmu sm program parite inject 57edaf37 rfn dfi unruy vilsel agen qhost bitminer blocker smg sitehijack waski bitcoinminer 12 vobfus grayware wisdomeyes 9500 16070401 nas dropped generickdz btcmine startpage nn neshta badur packer hacktool fileinfector kazy shodi graftor 80 m ac ardurk pkg patched l mm gator vbkrypt tool sytro krap pws icloader tc p2p net susp 0040f99f1 win64 f9cb8831 ci ht memscan virrnsm flood 1b8fb7 82 000aef511 tiny malware1 risktool 0040eff71 optional nbh ircbot oflwr tinba benjamin tfe expiro byfh wp 85 qvm19 save genasa r kcloud spyware linkury 84 tiggre evo installmonstr ipamor padodor 88 dr 86 83 87 startsurf 89 shellini 81 mira qvm20 aidetect aa bladabindi picsys mauvaise sl1 gencirc zexaf ageneric psw jr strictor scar downloaderguide 0040f6141 es bkdr et atraps th emotet filetour midie cosmicduke w64 androm viking inf z mydoom hllp rontokbro vitro qvm18 wrm swisyn 6 trojanspy mue gen2 a3 ab browsefox jc fareit pke susgen acqn aw inject1 gdsda zpack bc p vflood shipup ch siggen 40806030 antavmu at a2 58305 cq unm istartsurf a5 tsgeneric oat puf dldr infector pk ursu 90 cvf acb qvm41 zenshirsh flystudio q sl7 cc 22510d9f hllw tnega qukart nba rahack ait gepys unknown berbew onlinegamehack brontok vc dinwodgen mint gandcrab cs bitcoin 69310 ejafor floxif t c33730 004d48ee1 email dc dlboost dcck6 8csjk 14089 clicker wbx ale dos fugrafa trojanclicker ord dds webtoolbar skeeyah 39 autoruninf generictka malxmr recex 1c7c33cc 22061 ausiv sscope 9157 shohdi djtwta hematite bit 00481e511 shiz a489 16 24 servstart f3 urelas dpk nitol soltern mx vjadtre wapomi stealer kpuo hfsautob softcnapp bdd zh gen8 downloadsponsor ahnl graybird aidetectvm am 7b255d78 softpulse vittalia bublik y script hupigon u sn barys softwarebundler memery jadtre mikey dinwoodaattc cft ddos bcfz vbinject vh encoder starter tomc downloader7 700000111 selfdel dxtouo asmalws ojq cm blackmoon miniduke virux 3730 300983 aab r1456 40636964 multiplug hackkms f10002001 aukf3xgb 0052292a1 proxy 7 agentmb p8 eb557c81 snojan 30694675 sinowal bh dam fs gamehack 5hgpd5 dh kalr chinad 44344 ao qvm10 dm pak xiaobaminer 005256721 60 peed suviapen rincux qvm05 savis ko hw32 shifu 160 browserio 1051 virlocker obfuscator hoax cryptominer morefi nzq emailworm prepscram bas tscope nrf inf7 bi0cl3ei qvod bot a4 occamy cerbu 11 wh filerepmetagen eqqqsr zatoxp download3 n3 1286104 17408 nam cycler 0049c30b1 dp hfsadware 0040f9251 exddfs black jl al 22062 il jorik spigot turkojan avzx genkryptik dz zamg bancteian 40586504 mato 1388676 ccpk drp 6260355 qqpass sb 9778 35218 zen ep 30969350 smia vbobfus instmonster delphi r197446 rootkit fwwy pa t3 bk083edb smcf 30609045 02 otwycal exploit fvf 9304 nymaim filerepmalware cerber igeneric hh vbna 48 dn wc tp14 dd xmrig gf gm detnat tm 58b25de1 chapak a9be untukmu 5949608 wbna ngrbot 70 winlock farfli crypted adwarex networm ekstak 98 yakes ad fh qvm11 lethic 4614 28 cf qvm03 mupx syd samca loadmoney 0050b8571 5704625 chir ga smh vbcrypt pliskal zegost o2eunzrphue cz password qvm07 adwaresig regrungen stormattack 005070c51 trojan3 005246d51 gic cl tovw atvdd 98306f38 malware2 domaiq genericrxdx avdis ekl mr ssc ac7qzebb 7168 install afbd buzus smthbba 9999 kidep 002fe95d1 gc140262 gamarue sfyd klkgx 004bca3a1 dl lumer 0040f5511 jaike bywsd bk rbot s1 daws ae a647 swbundler fc rl amonetize ay ik vbclone 761138 cu a08a reconyc scribble dnsunlocker rmndrp applicunwnt 185598 xev pwszbot conime egw ew mofksys 8809 emnr isk 004d554e1 13836 ribaj uvpa a0squdf 56 virtool fu ntv fzza hfsoval ab3f 257 da as downware necurs 25939 genmalicious oci a247 tdss infostealer nbp 005169191 lyposit e4353219 sm37 outbrowse malwarex sf covus 0ad73352 tn6p starman tt bm 40270029 ejooci 95 plite irc kovter 24tbus cp cutwail ap freemium xta fc18000b yunsip banload runouce den ag favk 105141 installcube simbot 40542465 30955262 hworld 784594 fuerboos x browsermodifier kmsactivator 10 1801883 generik f20f adposhelgen 004b92681 autmine websearch siggen6 dealagent a8 undefined hpeg 7lv9go 30938733 0052b3dd1 196 74ax2y ck dld luiha archsms p2pworm 30720 rogue ba nc ppatre fakefolder sdbot 998 tescrypt trojan2 killav 30924928 sdld obfusc 0049c3e41 6332874 asbol eefb7253 dsdros eheur gupboot bloored cat ren krypt oscope ds pykspa 134072 mbt 43d23a59 optimizerpro 84dyy 36882ee3 af genome nyq muldrop6 vmprotbad 005205011 pioneer ardunk vetor 1841 madangel tufik possible bu ainslot f215f420 undef 315466 simda visua tovkater killfiles dorkbot katusha a846205f upx 17 crcf be toolbar multi themida rgu thunder fakealert su filelocker lolbot bner bn 0040f6101 97 32159591 appset r218196 0049b09a1 vbran jh explorerhijack possibl gc bqau monster dbpklq da0384c5 zedlaf 7412103 appster smt fg a08b vbinjectex 99e6 kn low hack dfay llac ojw ax qeo driverpack rc ah 222c6afa 1586 kmsauto ak aykncugi el shoudi onlinegames 177161 bbyk espnuv cridex a1db toolxmr cbh cobra 001 6450060 nha gh fkm 25 dqqd bagle bds sankei 528c72e4 malpack 9 azaj r110400 clickmein rdmk prng 43644 genericpmf vl 64369 msilperseus 207 encpk madang meterpreter aoyoyxzupns nca 85d93908 zemsilf 6113548 b3e8 hidp2p 96 00517a0d1 321947 freemiumgm2 pc gd muldrop 14 fxk processhijack gain dq sisron zard risk icgh 40223980 ny diple hlux temonde overlaynd fyig overaaand br 1140355 adurk webber rmnet 73 157619 adposhelpmf autoruner bl delfloader trickler 30504 convertad 4z809t 5988534 7000000f1 004c61081 wacatac kyxw sp 9b9c sgeneric agentb drop arrqo lx a165 hangup az shellcode smm aj 94 256811 13 smjt fnet trojanransom a4t kcmi muldrop7 gandcrypt 6136104 00529a861 tz lg bicone 14453 c64 jet fozt smbm4 2465 gameonline dmukv fm c425d330 gen1 37 rdml 003b8b111 kazaa speedingupmypc cryp fujacks 1847 dotdo arduk hc trojanx bb r230533 ursnif 61440 28161 6690085 dorv mods ransomware gk230038 dcmgreen a18d e3 54687 aq fsysna behav 87521 siggen2 r213603 43630 6655 4205b45f ncu wannacry 57501 zc pate zapchast qqhelper spnr bo f10000041 a36a fakeav dapato aly uds 74190 chilly 661c pemalform 0049bb721 horse 18 bfh rdn sillyfdc 3f7bd8 eex zpevdo cambot rlk4 0100010902 gz dloadr jmp6abneena 0040f0951 changeup dangerousobject jeefo 40371122 ebf374ab naa 40198618 mailru 5700356 visisig 004935801 qj 30 filecoder dwn 1408738 40886909 bjd ud ed bb4bc020 bq r12486 ghuxfld whl 121218 6333842 ayr8f2f vbgent 1759 shyape ka aqh jqum 00457c511 fkp mc 51155 zocpe apcu yu01bzbgwxg malob 4657 dlhelper nm fbx vx 1958089 downloadadmin 5mjfx1ertvd tn bloodhound 22 pajetbin bs mallar siggen7 gvkm4yfniq8 ircnite 77 ars firseria udu 4met7b smc prn magania iw gt 78 a4fa andromeda ecrypt bkbmt axj xdr yontoo trojanpsw 20191 wi radar01 oza cmrtazpjqxbyjwf k5vxezvhm5qb 2108 ceeinject modification matsnu juched 727619 222023 au 40697935 eah c933388 dynamer ramit rimod aqfo exp 79 installer ganelp downloader26 yuner 28547 qvm16 gen4 lc gofot bulz ibryte 53248 backdoorwabot 6665 jshdhip tofsee presenoker elzob general sm1 3218 bz bg wanna f10001071 poison pony fdld dreidel cuvt vindor 084dyy av dealalpha lamerattc bcd1d305 hp 91 jaik 497480 genkd 6712 downloadassistant 00523c491 bss bck pnjka2bf5mk allaplet 67868 sd securityrisk fe 0000 cd ewvndj 1866300 296399 esendi m4vu pih 197483 nsismod cswc gen9 c530 0052964f1 regrunnhc mgr mabezat unwantedsig fde 31051915 tn6k 41061596 8397 darkkomet heuristicmup fbfvyz vbworm moctezuma crypticb baaxi 40280435 ha15002e ffqlsb r3689 30855616 tmgrtext oscf 29 ja 67 31418267 dwnldr hpdefender vbkryjetor 230129 cpl diskfill xq5z4oobqmm fpng fcuxgn downguide wirus 2416 bwuwf salicode 004d367b1 dyre r233842 aaj c1453219 download kh avod 2x bho 697856 kolabc alphaeon 41072 mip4 3879ef autokms kav banbra chinky suspected hybris multios 23 d46e2dc4 lamerbttc a1 bitrep aac neh pupxax 74246 ser 285085 64 multitoolbar gend offerinstall d26ea68e pj 317134 rstdbjki fakedoc azid 005376ae1 unwaders fa dqqo rebhip injectmttc 0001140e1 74 duptwux 700000121 muldrop5 murofet aa42 auto bf smn 346632 005003ac1 c1bd2b76 655367771 genericr ar 004ba2051 fd 12424 minerautoit sector hz wacapew cky neu cb 43399 redirect mh palevo 178204 aems 15661 r230733 bundleinstaller 94c miras 004bcce41 ea k4n1h1k itorrent napolar johnnie autojhminer dloader gamethief cosduke 00073eb11 bj vm rh omaneat winwrapper kuku 868 tnq1 41472 gameteaspy ponystealer pt kms 260 ea1 er 00508e1d1 wm3 installmon genericrxaa mtb mn wannacrypt bjuwvk based dofoil bv neoreklami dangeroussig cv bybqne 71 gena 00523bf51 ccmw lwc9 pli if file uzeg qjwmonkey 005287c91 trojanproxy elnkal du hu 32250 9d4cab7b obfuscated bzub spyeyes 276534 smi kashu wordelloh sfone 83874 cr 8192 mz storm 9ad0 c3 a171 4aoup7 zjz agid keylogger ft triusor ai7tv3bi yil73szbko8 km cve d3709 31024535 bow 0051eaed1 0052e2a21 aaw 27648 rtk euxpvo ab2c jatif pupinstaller 6598770 pinfi kryptk click laland kqw nh 51712 7y5gwx elex overdoom tn6b rpv fhvj tomf pistolar pse suspectcrc xz 20234380 cripack ct cbaoi 75 downloadhelper lxrx an badfile 85679d2e 93805e60 potentialrisk damaged 004b87fb1 passwordstealer zum fam sakurel lwlv awk bd 0052d87f1 713537 bpa acea pluto puwaders debris mas hafen 4shared bpng springtech fou 01049 swizzor 15441 d1d856fe erazta r91566 wannacryptor scudy possiblethreat istartsurfinstaller 0040f8a71 fakems 700000151 dgzlogxbnje79d k8w 35e adwareimonster osk s2 doboc 200086 bmnup ali2000005 72 olext maener lt 125866 qgs 150175 cyzt dll orbus 147 9998 cst owd sirefef 36e dt obscure d26c1224 malxmrig fsh fakon ke r002c0gkj17 core uqexu87bnvg dridex 76 ge crossrider fynloski hm html mxresicn gi270001 so susppack npe ek 004c36e41 54 0051b9171 vp2 tomb lunam qc ps kiemnd mwf b51 ruskill xiazai downloader23 33 ccnc jd 158 ha220032 cg fozu r1005 8121236 9588 f89cd476 coinminerattc hk adopshel 001f4e2b1 0040f8731 renamer runbooster mlw bosbot jm 497577 naz 459 30609536 bjr lh trojanddos updater elkern 74752 bpxe swrort kp cj lx0c avkill oikyt 40533645 ea7024ba agentwdcr fdf a5bd 92 tsc 33030 ca fkfkjs trojanpws qukartgen hd genericrxdv fah kc 64c1 kudj jws 0015dce31 ildirim loring 1356499 qh gosys browser js patchload 34170 qiz vbo trojandwnldr a3c768ed a3e3b75f 1a15d390 0040f7441 acf79b92 005249641 4knk5y adclicker tosp cmrtazrwql96njyfz3vgf4p7v0bx 6753864 aqk coinmineratttca eq r245289 ato 4161 pjeb pornoasset frs ipatre ctwa pronny dogjis lz conjar mogoogwi 0013bf781 pack ammswzmi 873682 fbte 0048ed981 systro hqy taranis b1f cloxer refinka 435885 aymb 135206 bundpil asvcs3s 5vqtjo dad38 fw vawtrak wten neshuta adgazelle dealfly 444 ncr aenjaris 6221838 rmi vbex bk08301e qakbot helper 50583 004f5da31 aktv trjgen kj 8628969 004b8d561 mintluks 21 31 elh corrupt ewazfb adwaredealply kegotip cson xmr maximus oxk basic 6014 kyjb adwareadload 81406 cryptocoin 101 poly2 genericatg 31663539 sim foreign b2a6 tpt7 6723768 adcc 5744092 solimba 30606265 moneyinst cryptocoinminer 6448864 gg cn 0859 004bdfc31 mail spybot tpt0 31952 houndhack qvm08 doina adsearch asx enigmaprotector 384701 464 b4t1brai 368640 a7ed apanas hj lo nabucurobfs fakepdf packedent zp swx rubinurd b033 5rs7jr 1d1907f6 296427 csl falsesign zxf 66001 eor 0049d22b1 laa airadinstaller oa 4621 a328 34670 md zt a726 lokibot ab36 avct 084cuy inject3 47fa551513 zelphif x16gqfgx4 qvm42 apd r1968 40506706 aw0tlfli yy ij yahlover ul 31204955 0040f8bb1 forcud a6 ip tutu 2259 31801155 9997 sitehijactnttc besverit babar etdhkj ojx teslacrypt d1d45d13 93 25879 ju r233708 5898123 trash passwordstealera installbrain corruptfile prepender 51606 qah 7g2wfa toga 6940809 electron bi hploki downldr compcert 121913 siggen1 lvnl bza yarwi qvm13 up aco 8654 356387 me gdnw redosdru 005235fd1 004f3a421 esqohz 0052cbe61 ki socks nanocore 1396167 gamemodding flyagent beygb fuery fr 7d685898 nymeria nrc sm3 fprj 0040f8ad1 xed se 156 bbyo ac89f1cc 6781728 virutchangecall 320 eyestye kznr 61 71392 carberp 33795 smd mp 371424a4 ffa cpyi brmon 0051ed201 jz cmrtazqreqiweis9yfgmzonwxfej 15241544 6a606c0f nup 55 morstar ty 222103 el3 51306 000021441 sfydg ia quart r189313 ekc sms 134420 roc 51307 nss downloader24 53 0052d44f1 bdr 16638 gjc downloader25 loader salitystub 16412 ab29 opencandy 45583 22263 mt senta bat dbtjno 0040f9f31 zxc taidoor constructor 171 004be8431 ei 40476144 aace 32888 mf genericrxci noobyprotect aaq airinstaller eb2065bf ciwsuw stantinko 41 f849fe26 73756 abk vbbind 7729 shakblades ljdv r0w1jzoi 34678 ltch 18432 activator 9216 b04afd44 pxo bakc 004f50301 eii 0334 004d38181 disfa az3 tbsz cryptinject 009 auslogics hwocepsa hllm heur2 0040f5931 eltiw vmprotect infectpe ali2000007 click2 spr vtfodsvm 69814 5744094 a3b8 fj exypia bundlerx c199836 30602080 imonster 40390002 qksaytb qzu inf3 9c5 renos 700000131 tinybaron lmn pangu smb nxj 6717516 1388655 vt a6db izk gew oxi tsklnk xzwj7ocdfmu 30994065 1121966 40377523 xo hdc dirn bd49a88a hxo artur 13312 cloudwrapper 6726656 1854 004f8fbc1 2m6u4 40727948 jn auls bw 004d1da81 hidrag 40900421 30621520 osg downloadmanager 1355704 sakula zdpnbfc d139fd8 rovnix onlineio lhz kolab vobfusex adinstall 119968 lamechi 40360503 enwtvt 04c527421 1103 yolox xtz afc022f5 bavs xm4vdeppse eg 15 34700 eqtaog gbh 4yudnp d26b4d08 oksi atwpt chn d4ce 004cbc7e1 brbarvxyic8 iezf venik 4419 cbgs hylxxuc fby playtech 460152 a70b72ab spyeye fixflo 200004 virutchangeentry lw 9994 offerbox c8 elemental rat 326059 12106 genericrxat genericrxfp eph zepfod dx 005298171 blb 9a27 installrex gu ma qnt tnq3 28564 pupxdt 52 20512252 0051e2d41 0040f54b1 zaccess 471810 ed7 0050fa4b1 euh cmrtazrtfgjzgnulrco8jm9dywje run zevbaf gnje 52437 akw3muhi tuto4pc nni 20 lan 7761207 9996 56066cff 00522c7e1 1824505 pykse aaal gf090045 702fe106 0051ae491 nz plock 741847 smad 23834 6738927 0052d2471 cmrtazpenhvz4 8hodufilh37x4j gen3 d1d30e95 djky 0547 d1d88e06 00543ea81 foxiebro cqkyek 3151 muldrop1 cmrtazp startp d8992 220294 92278 ilcp 00541de81 delfinject hi vz 180224 gs wm d17 downloader27 cauesciyw 525 fednu gndz 172 34590 ccb webalta f9d51e84 installmetrix aa55 milam mplug aya lkqm dxlz r238368 0052df781 0052767f1 8c4m91 ed98 7wp9bs xax 4241 k2 8e7 2603 ky genericrxef malex 028 ej stormdos 004c6a4a1 mfcdafd ms04 searchprotect 84775 l2bs emtrum im r221183 cossta blhw a270 fiseria ubcoxp48wv8 887991 smd1 diplugem smnc qz 5756 fkdkns rmroudojrst 812 ab1d hllc exploitmydoom a8hp6rci genericrxds mepaow 00285d6f1 asdas 40509523 mswdm gp asbl 30823331 895619 manbat gyqc dnwsrg euy remoteadmin a0gi2lgi 19910 00528e801 siggen4 zerber rahiwi cmrtazpnqpospcheye bxlm6ezv4 mikcer 530055 bais acc vigorf fub trojanapt 135205 laqma driverupd eh jra qif axvdb looked 7136ec3b 1385034 mewsspy 41622 388260 31055803 aak tne muldrop8 qot 69 oald hpcerber coxy 5744087 eclv fpf fkc hkj unclassifiedmalware sm2 dukes st4 asxy7xscds4 1657 miravm bx virusorg rtlfz pwg0c 0053d2191 do 62 gunex pzkznanzdam woner multiple r230423 oqb genericrxdg execit floxlib atuvn agentpmf fakelsasltd1 detections c1118821 qvm06 237762 ibank ph 53515 hf blz 6232506 esfj 16132 ipa qvm02 005228ff1 autoruner2 c1708910 wizrem 6454574 00501e0d1 prl smaly 005409e11 download2 5bs8lt db9d32 aiw 40535276 7592665 foniad vabushky lntd mziy im6 webcompanion viselpm 230023 mpress llzl 51 pazetus lrrw xm igenericvmf s2280950 126976 dleuig kz2m 26488 64d0 gcuzf 22025 exe aa4 tpyn ld lzt r117998 my hfdg warezov 7iadub gozi vbs ymacco r243982 pm beaugrit 565w5t 1036693 installerex genericrxee nezoed zeroaccess r87521 d2d4fe f9777ae0 acgx 3584 ae5e 15d msw bziyas mod 397 30598445 genericrxbd r224787 specx patcher fmmfr5eyvjjib genericrxag hwolrd nf ndk bhuz dhcs fq lookslike bunitu 0048f6391 d6553 bqjjnb keygen lwy a30c af84 427140 66bf d26aa101 865 vj 1219 patchfile 5845 trojanbanker fo dtxv malware03 blackshades db eynpkz ain7cydi lm virus02 ga250a05 vbcr cur1 1109570 6959 548560 downloaderv2mt26g botsiggen cw s5 dys pupxbt jqvu upantix pakes o2ojetkjnuq olympus 00521e9a1 zydo 9pr3w addrop click1 00500e151 6335700 jai hnpgbwi 15gb13 chinesehacker 138716 d7 c1186838 smab 28915 s3863 00544e311 003523951 2eed0a27 bcj aa23 34130 b5bd oetk 671507 cuegoe 5632 nt xu 3010 b2df jt ks 28f290af r222077 a9e6 nl sgz doticp 2454 sock4proxy bagolod 64b4 aa3 lxc abf e6f1 6803841 genericrxab uvpc e2dd98 smartbar tomb00000001 8cfecc82 dd480c14 d8 l4ji 45 malware14 489 r162802 sysn 005003531 ctxu ic 105428 cakl snifula 11470599 40805559 dacic 1843 68 00516a1f1 004ce37e1 0040f57d1 smw miz 6519 xpiro jp blackcontrol rx hv 31348388 genericrxgi ql axve goabeny ovzo 7164915 d360c lgx 39956 kuaizip genericrxgm ntp bototer nar 65 asoecem ib 4685982 aqo 00001b711 e9c genericrxeo b5 invalidsig vs 2017 bitmin ajkq 66 lollipop 00001b721 18374 he tdvyr0rlqqm 00532d0e1 62713 c2452192 hijacker 168448 doena si edusxx ulpm sa smaz etuxeg hosts2 adware5 zadved 40203996 6c ir 40856beb wajam 36169 dgmv ocna fpmm ardurkhqc bcpa virtu nsnth s442725 dbf8d2 5f4 salgorea 1fhkga conficker ahmp jqap flystud futu darkshell eiqu hhqsieb jacard smr aiez enistery 5997 83717 002331771 uaws 20010 invader 6156801 pedex 30a xb pnfaoqa0fcb f1 kd heg genericrxfj r25058 d26678cd uk dcom meso 2015 0040f6bd1 ezdrwi orcus dcon cossder gen5 spyqukart 238136 ya 887abf0f r109611 j8b737iitwq yjz adfile ado 0051b5d71 gotango kuzitui ra 63 902 yf yoof dhwgp 6261194 buzy bpo 1150 di 7ov63g qvm40 fvz 84cuy r185010 ghz qm vg azlm limitail fiz 19582 wecod fabtlt 34688 alman smua 1938 ndm chz cvd auid en f810f ha150027 dna wizzmonetize blackv zbdu 8700 nemesis e09a77 cyix 005405801 34142 co yvbk 507f6e4f 31064679 panda genericrxfg 2de4d02b ggcu etthwn d2f7e882 gendownloader jvt 0052b8361 df vbv adscttc ux 246 genericcryptor 77217 heu glupteba khalesi 1495552 19209 7a4cc4b1 fasong genericrxbm b33c heim morpheus xt 65a44fa2 34628 ha gc1401a6 papras 34690 xtreme qx te 547 0147 cmrtazoekwvsnmjpmsqarhtxbz3i 394b29a813 pikor lovgate 630ae8b020 ftj 9827019 perite gmdb 5542 muldrop3 0052706d1 fc170192 a527a13e tuscas 31721678 21240 ol b870 0051ad2c1 hexas 40389995 107008 adagent 6tozo9hgfmohzgsbxbcp c2815101 dalexis kr ajkp 00510a7c1 smbd d1d8166d packednsismod yg clariagain oe tempedreve griptolo 11546 siggen3 mc6j 32 r213145 12138 lanv software rozena 004cd6d81 bcv0u6fi packed2 xtrat f29be0d3 403 wcry s4324439 swbndlr eorezo dropperx naf fbwv infilag hl fxzr klez yb corrupted 455 bgvo 28f27b9f fl a17f4dfc 30875256 other 0040f83a1 ngx fiy ec hvt r224731 wdfload 87f86ca3 rm snovir pbpb mw posible worm32 5744089 genericrxgf 5775 9995 puuv pb adp 60081 175387 34608 dromedan 22599687 ab4 663a bp qis win32file regvdb downloader11 ed426cc3 112 gunk 200001 probably 4mjfx1ertvd 343 1857 ile cryptxxx dumpmoduleinfectiousnme 2165 ddtmhv tepfer fn rp b033538d 229997 ly5v af60 locky fk 373521 bqa 1062 001f4ea51 ranapama zboter 30755736 eckn ec3b expl r115578 dhty5ux9n60 004b18b91 wxp but exmp brresmon fakefire kuaiba 6335025 805f1569 jftit 250 1658 smn1 rsj smy atfy solternpmf ncd conduit 4bc2e477 btm 230189 00516f961 overie e24c3c 3675a9 ywlbfud bmf 275 dlassistant 57344 5885 004c16291 kotver cmrtazr ng ardamax 406723 292885 632926 ckug r221632 apf 0051918e1 d19ab5 blk 40815110 004d4ed01 fi iparmor coremhead gxn malware09 downloader12 r217833 0040f80c1 ss multidropper pcarrier bcig evvppm 004ce3951 aza 64bd erw 48672 crh 0040fa371 akc qmlfrt0j91kid zgy killproc la driqkh r002c0caa19 34804 qvm17 ayqe 0052659a1 ez exfelt r219415 7a361049 qa d1b7e9b aia antiav gm2 084gyz 36100 fb fpd bdoor 59 ditertag smcp 00176e371 cjoo r223605 493 bitminerttc cqkksp a33a727a mlwgen softonic 35e013 vtflooderhv 7lnbtm browserhijack dkwt njrat 6170948 yandex bcli 7gzbs6 golroted 3l1r83 1388682 ho r242806 6260333 a479 7k8fmv jyb 34058 e7937fa1 acz kelihos bloat dia 942 fybx ethqtc htzb 5471 extenbro 10333 4ab5d27b 50 gj genx 2ee348b2 td mandaph 7o6bny 34236 dgzlogwtuabogjmnjg bqzoew 004b75071 fwd 420 loy s4530269 npxzaec aa79 fox thecid mudrop 40727988 1831 smtp 21302252 d731 r206549 004d80a21 netshta r229425 any1vm ojsz 004c42fa1 yu eyztvm nvoe pi zlob lyq 31031667 btsgeneric r174475 56xigo eh210001 banker1 avyl r203456 tnpg simfect 6980759 167936 cosvdb s2399322 35858 ev 6acb9b9d fls qvm04 jtlp 461 alpx pwstealer ats tiqd b072 luder angryangel 0access gb140013 sprotector 40526563 smb0 10b0cffd cxoj threat crypto adloadmttc 1mio9l nbw 2547 13656 wali fnaffc ww 4f4 firseriainstaller pcchist genericrxdq jmo ze 004c311d1 106 d4d04a a89e cb4 32670 skyper faq regsup re dhcr 2d85 skypespam menti wz 100891 siggen8 003ea5181 0dc56c850d chipdigita3 320874 downloader10 lr by fp9hj7 yb0y fizu 1z141z3 ohc 20629 inf4 niv d20bb8 7554630 og morphine tv vjh 34266 aiwl oxm a4d8 004d820b1 nakoctb browse l127 m1r5 roue 9e10 tpqv 19 driqkj 93327 msfake 0040f7f01 1030203 1ba837 wk adv bocj rodecap ov 3424 kk 715 dnschange flxpki 1388662 ga250340 004ceeee1 dha d1eab767 260379 3301398 51669 bakb mdk 8bx67dehxck 5ecf618d noname fgau kpup oncer bhd 80345 6923327 114 0050c6b81 winnt genericrxer kz zz4 6055402 004c11fc1 eyu binder r213772 6722904 cnbuup qbot tixkc eakbir elenoocka notavirus appl d711987 700000081 jq stration cryptz auz 134015 bga cpn 005378b01 37eb7u cvlh 34050 s2806265 1576 acdkucc pulsoft 002401471 lipler kukacka wenper gyepis amjycbfi 004c20af1 34790 1607 r250830 140 139350 44416 005035811 vho azrzv hrs mqqbwgb li 003c84cb1 411 sigriskware 32590 acf 33222094 r233450 neshtab systemhealer microfake r885 fgk wrhr 5462 f433 bcje ldpinch d1d7e080 jkd 1139483 4ff megasearch duptwu massmailer nud wid 792480d3 adwarefiletour innosetup d86d8afb 6297788 du8 a7cf arac 35452701 28160 mqe bsymem 0055ca211 downloader19 smth alzxw 1556 eznpmj fphl 42 swiftbrowse 3b0f4bc4 9e4 6022028 malformed evafmt 102148 ea509234 loorp lockscreen 1216 tgsa r224901 17867 9cea no 523 40267082 34692 1fy3nv stormddos 97a dgzlogxnvfqqec1xba dirtjump r27090 owhx hakuu 450 fesq 24384 hn 5jifhr aav tw 5969619 ekgfz accd10d9 gmf hb 001ab60e1 5c5lsj 5f0 9e03 ae0a bb89 i0nmumnqepe pg ha190043 s5304897 fda drolnux wormx pandex 6327385 bplug vbpack 6334882 zv 7a935125 210488 appsetooo tp mauvaiseri ru 31242302 oiz b4 ada9 maxim mfpw hy lilu imali 00575d031 lmir pjtbinder bop b34f ogimant snoa smha 004eb0881 279225 ddvbuad dgzlogvwdkfou8cszq hx4cepsa hr dustysky coins 6418983 34758 generic38 r217167 65fab7c9 c46e6d2d zffp0cewxwi banito 290 d632 286822 dxs xbdti perez 6652 bfrch turkojn sln daci nlq fesber 913 40673609 s2218022 1603 1egj5j dojfap mocrt genericrxgo c148121 lvib d4e99b for crack 2753 kpp atg agqr 004b75691 004f35d41 backdoorx 8047 xv bladabi nbx derdero ui nwdz4shzbcy jw erozmu specaecupa sodinokibi 74386 447 fckb yer 0052170b1 005293581 pondfull 286993 aal 460 kdv ys cy 4299 f29wbhjhjqa 84429a styes sg 132 lunastorm almanahe ewrbum wof 136467 6804088 avw azden aijw wauchos bepr bfsu 0051b4b41 manager 00129bd51 phorpiex hotbar a3a0 50933 r188188 vgeneric mjnqwy09kn0 bsbot genus xhqeioyky4u s4627646 1034445 chg d2b409 6804140 xbt gat 4ff147e2 41515 57zyc0 eo oskb bgfhm nwabi efvwpu s1979170 ee genasom kazoa 75135 7h5w7g installad asprox 00532f411 40956228 s1656376 s3755 54464 b180 abl ascommon boav 005104df1 vbviking dvsrfy 00524cef1 kpuq 54bf5f47 0710858119 ay6felhi r214290 a140 sventore 62112 0040fa0b1 malagent 00529a881 pef genericcrtd dgq a02 evlqpt elvo nry dialer 86303 sillywnse pupxed tweakbit nr uy 18454 smal01 482569 a7 r242738 swz 435 614 18592 allinone updays fjybzr f016 flofix 005 f11 flihmx b90b cb1 e60d misleading 2224 c5d22341 io jzzv b4ab p10 xdt ibashade 004c1ed51 004bf10d1 9515 optimizer bundleapp iftf eggnog xcnfe rmj 3539 autoruner1 floodfix 95d 388387 fakefldr qpdownload vmlfrbwxs pib 45i7 0301a0e1 bjqv dlsponsor 184320 00535f0d1 ne mtbdpqaueik 9827000 jy blueh anu 30854925 247110 21257387 netsky jjc ges tppb 142139 e0cb1b84 aap stormser 59269 2222 winsxsbot cuv 542 vtflood 8rqmvh 27 122488 mmm 84gyz qukarthv r002c0ca819 ejuhih r245021 363220 1clickdownload fakeregt 7mfe0x hh3 rsrlfghi avlj 58873 lv5w 00110fc91 na smj tosq 004bf9011 6330434 udr 7e2a5275 supatre 950 7238 7xj4ru 1003 lf c2864099 1xqow5 587afbdf pro 004993691 d26fe27d 876573 0053bf701 axyza socelars bvrqhu startb gwc atdb 291849 ddxz eycikv gb140015 59ea smmr tuguu sm0 aae kanav lkp b395 ceram tn9h arha dmqp r221287 innogenforcalgo 1ab44e89 nny aeahd fx faac dqqr xaq ffi 002935fd1 snarasite ae162e5a cassiopeia 51917 115 vagoto 74200 jxcto it 47063 xema fcbadw b3 windex r235056 aa22 rwx eizuzf r201980 fz aupa genpua bd9a8e13 103 005195d71 247287 854505 58296 a11 browserext cz1 c593f779 6677 pcclient brf ajsg d00ac9c9 erj ur r174595 smd3 004c69561 e0878d74 xwt 824196 471 ali1001008 downloader22 9984 ub 01657 fujack pac eyg 005224381 dbycns 52876 be3auwune3ii 42tt czdk d79748 cmrtazpvss4gbjbxlcmgrwzedow7 qgo jb fraud spygate 85702 1c1f0de1 005376ae 6717566 a7c6 smssend rontokbr 33021 vundo qvm09 5478798 parmo r37539 41167 23092745 mywebsearch eztb rslpttc f800 eccndn zbh 70344 gh0st virtumod r248696 pupxdc f2a4b9c7 6euzj1t 2zg 4255120 awc 30925132 g6 backdooraxjdll urr 40370898 rungbu inejctor 34132 ajr ratenjay r7666 innomod tpbq a7b53705 78584 tjndroppr 0040f4dc1 82feb89b 24429 005153df1 7930782 pepatch 1299 1141 dv p9jeddmkq7s e743b39f r198135 systex 3281 cd850ca2 ratx emjumk igenericpmf 32570270 cardspy tenga hs s19661368 injectgen main malware08 80182 d26803b2 daytre netfilter 74245 mkar gbx pionner noon 2qmzzjiynhs c195259 cwpjcly5u1k hbb kfrsw epo safebytes 165 powp ahj 1901 s18904576 kl cola genericrxao bbyoa download4 40459970 slugin 1312660 6717398 61286 letipig t5jx6ahnemi yb5 s19176685 runonce eraluc 19227 hfta dlnpqg 1c92c6 0050fef41 pswigames cyqd fea996a8 34634 aum ctk em prifou 000141f61 dkn 2904 781 gc280075 004cf6b81 1dbnr4ngtu obfus vbcloneaattc 58276 aet installcorecrtd 34218 linkurygen ii flx iwjucbd 005478071 trickle oficla kw 34254 6169544 antifw difjwy enpf d5b60b6 v1 canbis hkmain 12a d265c4ec vbtrojan kec d5f60b96 99c6 cvflp 63527 6956 30272 c4 lmnfaoqa0fcb mybot 5594 bzks hpgen 7700 csyr 22593 x9akhl8rmzy r86937 fy mbsx r251631 hbyc z7dbem6zlnm tu r127004 game 7lng48 2be1 nemucod d2728cdc xeg 772 ey s1796222 eczy epack elhoip fhh 31656 downloader17 ergkri 00532f5d1 lineage sohana 004dc9761 1jgkljwl2mg5y37chmfi 94583 benjamhv 108 kzf cihufu 7oybry f42 trojdownloader vbagent macri 9be6 vba yx 004a91c31 29121 vv rubin 15040 saf sml lnr bjlog 0054256c1 avfq 1500db9ce7c2 sybd inject2 scriptkd pupstudio resoric sq awfy sugar 236 5qbrk1 agent2 cmrtazruuxkvfevujpt58o fshku diofopi ccng 64504 gop 005290891 a70 nu 1203 r88085 c65 86875 22030 dwthyt 192217 cfkdy bsm xpirat 6645 sorter gjym mdrop bcx urelasttc 116287 dplrap 49664 73f jo 2138 iona 71011 32768 fdja nevpvs 11534 34780 00024be71 cmrtazq07p3xeejqlm4vtruumkzw wanacry dj 38476 tinyahqc 32322287 d3eb2b 01 ewrbyv lpamor 7n2jyb angel vtu downloadmr pqif patch oj 2f2d89b2 speedupmypc 47 dsjw softpulsegen 9978 yn ab8b 61d9d608 aytk smw5 r45219 3d4c7 spesr r131999 s3376907 b577 yh fdgo korat d1de0c9 cmrtazq tas ns 005393141 418723 floxitnv c638970 ceeinj 004363fa1 add protux g4oelk2anz0 004bbbe41 www ekw cqst 9cec0 awm genericrxcp ggz gb23009d philis 548 40299516 hwmaepsa db1a43 aa9a sy dloadsponsor np dbyqsn 1349331 d4edf 1032151 30871431 beebone czfv 91e93787 cwf sysvenfak 23214054 r205727 lak4 qu amvg c427373 tnq2 b29d s23068 fraudload siggen5 7izq97 gify pincav techsnab screenlocker aidetectgbm 34152 cqkilw aeuc heyr 7073 d26d004f 22048 generic32 917133 linux 004c41721 000043a81 r1452 ransomcrypt bpchjo evrlju r238376 jorikgen r229238 77ycil 355641 34658 smaly5a ggf de be244f44210b didxqy rz c622804 resur pga 8272 5120 cqidpv pahador s12799 ayuw neobar c27 998b 511 00013e901 94640 322565 bbbb jk abq b2 gc21013e geri ntivakd fareitvb ceba26ae ii2kr0iv5ci 58537 pqcoftphnbe installmon2 smzt 55028e54 6489152 msilkrypt02 aqe machaer pwsyunsip r758 nwl 1821 capsfin 1992 56634 0000439e1 genericrxea ghv bavt guarder 40219434 dafaiy odmhweeucoi 003da8d71 gq vosteran e30d4a 366166 rgzka 191268 aipg kraton ayvz b95 43 drixed nitro kjbafqc css 1b mona lp 370641 bvm 005452be1 fmy dffywr 4520 dyreza expiro5 randrew nq fc040000 d157ee2 0054d10f1 angryel fpdfnhmb tnrb 77476 10118 19961 bifrose 34144 0040f3da1 r83549 padoravm r198137 gv wgfd abzf 3842741 gamevance 005104571 dealplycrtd 1023945 sg1 34796 amh stubrc se30272 r228636 1988990019 7ohhmx 00545a6b1 zvuzona 567 11473269 cheatengine muqp abfq sm44 installopt 5242 fwko 5896042 71043 29002 ee8ba259 genericrxal 49ae 30641 00526e411 mjuu 004c603d1 shm ar4 tnsb gi c4d8 mvxy pysn 0191 673fdb17 rr cdby r160937 00012eba1 001e7bc71 uwue htau grenam bjz nsanti 2398 privacyrisk usernamehuinoga aul bwpxnc d19ff zo 1gs3xh vbra agent6 qrg penzievs xp 700007861 qr4 87540 4nuhm kuc kcsdzzc pziebz4vjce 278070 junkpoly 872 1qraug wat blacked abs 6723905 asuspect csnmkc bcqf aiou hosts l3y3 r228548 gnm softpuls rennes awj ayzg ed1f4ff aymqprnb wzj fatg csajzo 03012021 a236 4410 4160 hesperbot r119347 30966789 0100420982 mqi downloadmin kido 6561 f912 toma 256375 malcrypt nct 34136 yakbeexmsil 6323528 xfr dhx 15448354 r59840 60919a19 zwh nyb d1c7a3c 12657338 fjvery c9457d4313 byv hxqbepsa a603 12494 boz 4480 c2571733 cfi 004be57a1 3j4 v354tmy 5710308 tgenic trojanshifu dbzksi qw 91136 p11 bbce54ad 1903 nmnfaoqa0fcb r002c0cge18 kbxy 005191521 7255 tpd karagany mindspark 34722 genbl tfl bbsq 472854 nzvosl5th5s noancooe 5wak4dhesrq 473524 1388659 1314325 24c5b6f 22515 instafast 6944 gifq 25891 cryptonight 1a bld 260835 401574 durta r203929 dqqn 7xdbut r242842 99941 6787524 fptt pdp bbckw 0040f5751 tg 311296 genericrxhc miuref d1d6d1c0 30611111 qvm39 oxypumper decap mgz 150598 hijcusp jx gjn pwj 6840595 ro e01158 dorifel widgi indus 16b copyself tpcv 7150 blamon 4639 xmnmuizsf1k 6804274 avqg 57235 r131885 qvm202 wtk softonicdownloader tjnransom spatet 7bdbct 14386 5819 0323068d 126 2016 gamania 1474 102 fusioncore payload r164118 354681 17914 44 9976 p9 34126 j1 004bea931 ransomlock 9aad fakeusp10 mv 1787437 vgz 688 7vajd0 004f5e8b1 364 004b8e971 lxev fv 21504 fmgk hyodepsa r7826 r238850 tnsd bayz ashb 34760 162 bvdt 25751 ih gen15 abd cgmu licat 87685 40802363 paa genericrxgg 526f724518 kitpyrk 52157 ffvlkl bij lwaa uc s1829144 blacoleref 667579 a43e 6726654 180098 4150042 xwcdln x3bcu6yu7db ef 004d38111 af13 r220228 5xaovq bye sheur4 puadlmanager psw32 xx zd 0053305e1 bbp 113 9983 ransomwannacry 9e2c ril1j7ii nagram dorpal ali1000029 d36347 6838244 327680 000013501 ixgl nanobot 308862 dpvlrd 2938 fckw 46 1zj8 dy 003c363a1 autog padador 15593993 2773094c tn6o syncopate fach 234 51371 fgni component fakejava kmnfaoqa0fcb 1037412 koobface atbh 1101341 bwm 63174 fnws 307200 c546070 al8 111744 153 ovyq 43324 ae7 jkbp cryptolocker 622 5dee1ada 34088 cym cmrtazqgjfzpi9jsqzrdgrihbrwq injecter d1df679b u2 jnaq afas 00543e461 killfilenhc 74270 0053ba2f1 genericcs 004931221 kypu generic37 25355 8944b731 asxk bqcb dc879512 ff 388317 dg crossrider1 genericrxhb r3593 afb72f0a tnqy 6637 azjw 58368 netsha kx xjgj 9aa5 f10001021 od r231859 81304 kwbot a1a2 qqsteal boeu axzdc 43072 lagi nthook nj svcminer swe russad fmwz 001fff681 tnwh pornoblocker qmnfaoqa0fcb 47483e8e 40205070 xdc 34574 exrnic fdcsz1jhxdg fp cssoeg dk cosmicdukee 10b0cf31 r251968 6911718 135103 pz 4268 pupxeg cosmicdukek 36753416 70598 342915 cmrtazpejgowwg6xjslrpnj0e3hg 138653 optimuminstaller comrerop 004b203a1 292710 detroie 34110 0050d7a31 zz8 0015e4f11 qotw hc050083 d266a173 wintrim zz5 ftmv 300 dma 10b0d00e 2f cmrtazpr9a7xm42amjjlw o1a 2662 8798 preloader murphy rq 24096 00c5bb920e b7a8 004bf1bc1 0749 tol 142336 grp 6119664 kdz 0055c5c91 258672 eu honret 80492 951209c3 30497337 cpsg xnygl5sjvag 1036455 ha220005 efc8 aad zz 1103295 2vcceqcwlec 12429 hc060064 12495194 d485cf 148 sohanad rd spora rnb9gajgu 0051724d1 spyrat clovis sl 7983 chs vbcryptvmf 230093 accv soft32downloader kq zs 416f5770 d26561da qadarsrz cgitk bundlercrtd pr 540 05e2cb41 00528cb91 10749fe916 21f acff kespo 20716 socstealer vmnfaoqa0fcb gsa 359339 enigma fwqg lpf3 9594679 f28ezjj 2eb tibia r71203 grr m2bz 005464371 sural d1d9d08b 1100003 cx r122430 mentiger kf m9no awa 34686 lml 6145 genericgen 56716 cmrtazqjlywyfbrauobhvqbsfe r149627 aixef scarsi 727 pestaple fakedownload swisb r01fc0dir18 22257 tpm d4d6ce mgn zloyfly 12799317 hxmbepsa xag alureon r97433 laqx 0866 dzermhxsycy coc 157 176128 cnpeace lxi5adqxowi 9475 99db yoxuf4uzklg brsecmon xetapp r19970 smhf xw 0016c8f31 vjy 40318424 mediamagnet nva 43784 aow d20cd vxozt dropperaattc 16148 r111396 251271 242119 24492 6552923 bqpsur 67180 sdp 27105 nag genericrxgt 3644241 aa7e 61631 a4378134 sm01 84f6ca91 62464 dgzlogxedum3p6a 31145 vunspy fccb d4599d faif lha gmt boc 004daece1 chihack 9934 tnrg fdxl gsu a8ef cb615545 s1776232 dfheig torntv 40742974 wmnfaoqa0fcb msilkrypt 43980 hbw gb pupxag hygbepsa jv bsv rpcdcom genericrxba cstqaj muldrop4 genericrxgp cuygfbc 9977 pleh ausd 64d7 d3036b deceptor lrddieb 7j jebp 1c2 003e58dd1 ano bdiw fme 901 11432 3bd kb y6 stormer adgazele tole exxroute 6aa9 emogen vbgen ava bmgorw blackdua delphiless s477658 d43836 hupe hsu android apnl genericri qvm mo downloader9 aoqbylh 30758917 1507 dico kt s19016571 azxx 9972 m3 125 zv4 vn adwareadposhel 65e 9982 brantall ctb ncqlu6vy truko r148548 eaeozl pikorms zexae 36 hpgandcrab 27600 aixeq fib 003fa0611 005346871 xmnfaoqa0fcb qe am0 58 ayg27pcb 246115 ezgxcy eagnph ffe4 d155e1 6804092 0052168b1 chur b4ad app cmp ea7dc14a s19227754 lolyda jjm ncttu 005170991 0040f4c81 or jf 5129 expiro2nhc 90650 ajfom cmrtazo ribajgen aaceh rlpacked vbobf c164094 00519f1e1 5e7luk nd sml3 163942 sillyautorun hpemotet fgy bsmz finaldo revizer a343 ugub 6313787 umbald 21252 086f e6gpcu startpage1 49 ji iq 99dc 9933 11842 allaplegen 0050aa351 e2c45e dellboy s8447 57 3ypg 004991e61 czp hllim ae6c aaef celz delpldr 208896 dyywqt cert 0413 wncryldr coinmineravdba tnqi 273909 asie 87550 757928 koutodoor adwarelinkury eohavn fhk gl 12477 ave reputation 4pc2ok otwyacal s19361443 hesv dnizrq 9283 el150010 eb jiq gj27003e 004e46c61 tnsg ga2701be genericmydoom 0054ea9e1 bhzka d382f1 yy4 1038 000ff3061 6261684 sv heuristics b640 9230377 aqvw 64a atj d2b81c smac 11607 qvm01 protil jpg lk0q pv enwwte gzc dmid cnhacker xc bk08494b processpatcher yantai 30ai13 m94n 30828811 gzj5ae9mgsg 7000001c1 230040 000537701 004d6ecb1 004d09981 hgz 178208 kadena cupb 00528d641 amx vigua 49851e96 suspiciousatg skintrim wl hpcrypmic flmp bmnus dpeq hyb deotvwvxolu rsgcfrd mlite lite badfuture startpaget elenooka 34742 d20 krucky ec502958 7737 5vaur4njuno bagsu 00537eb21 6722806 c4d5 2386 zz11 b564 06d41943 47c1ea66 loskad 004cadd91 ut gk lzps ha010038 fcvpqt 139 2ho5ur 9968 yzy0ok eb3 dkxl salload 0040df0e1 fjb doih smalym 72354 apnj jaiko 00511ead1 a89f gb280010 30865232 c5130b88 tiz kuluoz jiomfn28tbg zzz gw ae75 04c5273d1 r219968 qn apl c2357910 s2924937 dee59b718230 102336 autoitgen kles b30a kz0n fbrg ck4nt d1e325b3 qvm23 lh0z oc cybergate bxafx zu genericrxck d47809e phpw 7h1cjp harm fatobfus ef4893ae searchsuite expressdownloader oh kro 56727 2196 yj smla 586689 34216 003a874a1 anfh 1471 ffzl r250945 da987 jcen 936 hykcepsa eiy 6305879 r37162 machinelearning anomalous rs 6495 ek24001d 56a smnf 104585 9770d145 fma smqx 3555328 2b2e bamital rund r1831 ce0719ba tjc 34294 s2037912 179 a74f cbi cva a0b4 biwun1hi 04c500771 userinitwininit lsswh bitrepeyp 28507 down 004f4fba1 31713010 ammyy pariham afq searchsetter lotoor infected mda drd oib sigmal fqct mb qqlogger 005223711 gn cmrtazrn6amjjtb0wusfajovhayt genericrxdr vopak c6 d1d96597 innhca kypes 1220 arw batbitrst genericrxbf confuser wt downloadsponsorcrtd a306 6775202 icw ftxi ennfaoqa0fcb nan 004f5cc81 sme 6loqslbfumi dnej 230125 numnul 11523 32298 s12202810 1775fs 30831489 rammitnna 0052ba791 1ptxrq delfo azwh svchorse de09d ci1erp5zu hmn tugspay bbli ctm 291177 polypatch 155914 ig opwh fdow dcgupd tovus ganel 004d37231 a4yj0w 9b80 downadup 230508 aovhryb a0672e3b 22020 2424 chipde genericrxdo 1033829 1qrzi1 0050c6701 9e49 jed oaf is r234001 4bdezk d077073a umnfaoqa0fcb 005426951 host npkon redcap 0050c96f1 b576 10b0ccee je krypto nlp 23859 bzbaukwkdpb 30603136 gr baox tmnfaoqa0fcb 9789017 36864 yotoon aug waldek d24a9f uqa uu 0053e8561 35392 upatregen egz bk084025 0c86562a azrvz 148839 nxc braininst 2644 220566 f596 25407 dpvu jaw to4j dw pasta r14017 22009 phfxqdo5qac 00557fef1 22016 zq b308 1633 dishigy 9989 2e664af2 1ei64a vsn1fg18 hg bkt afz silentinstaller 9932 innobundle oqrkauycvcs 186443 b0cf 6abc 137830 mj dba wronginf bxj s4396331 malwaref hff amwuoini 004e34051 40499978 6838221 emys ba59 04c4d93d1 6793772 005005451 gqj 305824 498 qp5vbguof98 d7a5a85e amjh2thb 2865 6376318 chirbpe type rasftuby 238a4041 36568a 7cb72385 0049f6ae1 ydpt8kfwpke 134b mnfaoqa0fcb e31788aa genericrxgv 80w0gp du4 1381932 tobfy smax iat diztakun 6726814 d797a9 rb d5debd avkiller ransomeware 43882 d1 6804628 005452bf1 cmrtazopigtyejwnofp5bjwylstr 43372 in r002c0daa19 yxi jj 00012ebe1 kbot fhoixn 00533df41 lm34 darkcomet 000fc32e1 genericrxdy crawler byvara hc050144 fearso ymnfaoqa0fcb 30879964 19514 742522 005393151 vtfbnttc 6417593 nk adx 6021842 7jiloq 6840779 browserfox 161 staser lxua bzkem za 1etewje 3127 hwcbepsa qqr lk genericrxfo m9d1 27f genericrxbc 2e7 lisw jbdqjyjhjdi 9xeeiplc1au 1449 spv 3ddbedf1 acmcl 232897 189562 ckef opensupdater cmy3u trojandownlder downloader6 downloader4 slenfbot r221409 ybi knase ha190016 ni vbinj anua toby cheathappens 1362662 39bf970f aww driver zeus avz 80617 xe em0 55f5e717 dah ransomx cloudguard 004c75411 cwhd bulta 54b2f32f 0053e9eb1 354 83067 939 gate 5535755 icedid 138181 fc180244 comet zr4 f80ur twbgfmgws0g b99 004bb7de1 cws r154407 dbvlfz aa95 ziy x1505 kirts a8ovoiii 2229 sform webprefix vbinder d134c08c signed 40528594 dild nestha 6bfd05c6 cryptoff harmlet ejqe revois 7152522 9964 mehz 0000bf9e1 5834 6622765 reveton 24502 gfgf 5afb hwubepsa 9971 genb nkq 69a2908e vu 4980865 jrau aef sillyp2p svi gentroj ghb bogent 22562787 305923 105 210164 38 adfc fqd 1939 obsidium faedevour monitor 7y59yi 537 robzips d2a266d3 g4 3226 r029c0odn18 r201779 9967 7151253 9a7 hla 40441877 12800 ln 401 6520432 276 foy ahp 421888 5ce 189427 988 qcc bfb loki 1565672 0027d3461 cojbpd pupxfm eh260244 b53c 005497bb1 lodwondiwmv s17686708 6717505 d1dc265b 005234be1 6731 abh droppermultia genericrxcl r4378 315 005050991 amwhwxbb ludicrouz 0040f6b31 00556e571 sr lohv gbd genericrxen 004d3edb1 mi akz facv 231 startpagesunlnr 838 004d95331 smkv 004e497b1 eixaakmmyob 281969 mu id qb s1782204 ewhhaq genericrxhd filetourcrtd f1e dcomrpc 34662 bebr vik 124354 39201 hds 63762 31317 epnptq bgseraatib0 a1c cmrtazormzc8 8mhluuuwo1zwp21 gvkbuyfp virndmc akk 779 smmut gk150012 kokr 5gzhfo 34570 7151250 34282 r18258 otran 6b2461d1 buterat cmak cijl genericfc a1c3 vq 147456 yw 164420 daqc 6598201 faww r233980 covuscrtd 002a8f0e1 azqx brlvjf 5731517 525695 qms 54110 smf sfx injected chipdownload cag 32187003 32248 00008f2e1 00549c091 bblfvy eb0 nck 00499d731 0051a8ff1 rap 71077 czmq zjkf n81 jgz annfaoqa0fcb eimhvk mznn akxd pyf 8de7764b starteryy 415 01643 b38d serverx jpeg fq3 luscu installdrive 3114 axzd dannado installmonetizer cmrtazpd axy 17937 hax inbox 6mjfx1ertvd toee 30cu14 1117983 giz 1178 in19gpqule 12227228 40795849 pkprn multipluggen pwsspyeye 4d0f2c24 fd5 30884441 23092731 23076000 shitovervbx 9689 1549 dqaa s3094761 fnqayj browsefoxcrtd s19852065 cmrtazpgyqnrysw795nvt isnieg qvm30 buzuse r25024 rush 6335034 15599212 badday 00518bb71 xf zdtnkjuu39jmewd dsoy 234d gfg cauqic cmrtazpxyy1ao0jyij4dhcosvmo0 gendal pasista hpbladabi sigadware lpecrypt 40489 d3c51 corruptpe trw downloader8 euparm mss stihat qwiffa 0050cb4e1 gpz cdkr aeh 45315 ridnu tjndownldr bruhorn r217004 111364 amp 004c2bda1 bbei 27bc0672 s2043631 vox oyxu r248454 zzyngcn2v9c genmaliciousa ppbb 16461 16696 smxc origin aitzatci 259 75lgdf tovicrypt ayls aat frauder agentbpmf xdq 5500p 31172 cjub 72171 kntpsoylvgk trojanp go bundlore 230280 smx2 49726 yusp zdengo genericrxbb 34738 tr4 7iqa3w 1252826 1104226 1037401 ly exeheaderh 149 25782 26 f1cd 005239b01 0011d0251 czx 266 00529c641 b79d057b 468a6143 abw tupxoeip revell r002c0diq18 viz cjerhf zg 3ecb3dc0 smo aspm 5gmadvppfua fpi new 21683856 00492c871 a7722018 filefinder kuaibpy avy amw 53e 91282 genericrxgy fdob cnar 21502 lb cryptdoma guildma a918 vigram genericrxgr tjj fraudpack blgi mg1b eeq ayle8nci sm4 dermedo 0625 dnq 6015 fpvk extinstaller azorult 00516f dnschanger 219 auki 313093 silly 3666 0049fca71 izj ekwnkd 5710245 7h5wha 00522d561 1108484 mta 7xingipiq8a fav aqv dqw mlwr ndw la0v al3 ef2b 9981 0b5e64ab a0 9a13 pefuj1 27746 genericrxgl mjel 004ce6cb1 yah ehuovt bbi filoskeed 0050a9591 318808 df13991f 005450931 10860 21257922 skorik 9993 9991 405 50691 b73f tm0 vbcode eralss downloader32 cm0 macro r228418 005087911 r002c0dac19 tnub gmts 00542db11 118 8537 mgdt 200002 m81 464617 r288360 6335648 g1 34770 erajhz s2722584 005235491 asn montiera 4848 21506440 reptile gael centrumloader cnnfaoqa0fcb c2041432 miancha gavt 62388 smnfaoqa0fcb ha220006 jtvn d625 77824 boh 58e862b3 adinstaller 32251 d7beb94 36780 fmobyw tomb00000002 qlfo 5515 dmnfaoqa0fcb akjxmani genericrxdn 2413 7yqqhk pf gen0 gddv 7903 0043a7501 ikds l8ew hebchengjiu olgames 183921 004c76c01 c2471776 40395619 84693 ebdq genericrxcn csnpye ie 497 clcu 278528 0054828a1 d7cd0 bizd 9c28 crypter malware10 gf0701c8 omnfaoqa0fcb cqvnru cf9259cd base brsmj computerbi nax s19011062 xbs unq 38732 rgwt 0089 tg0 48948 150 dgzlogunxy5qtsckrq 19105 5901233 se2 hpandrom smt1 udi downloadasist 2493 71058 barl 6b47672c smck 2ffe9843 boi cmnfaoqa0fcb 61463 40845545 webtoolbarcrtd 12432 7yvxyc 1583104 qq 4303 ogz 895 r197444 131 r117522 9985 bin gc3101d5 04c4d8841 r229668 tlm 6336261 deqiht aiob r118784 r211898 005524301 cmcvwb hovi 0048ca511 c828b5d0 r190826 xi b352 vban smartapps otorun adojnhc genapplication 154 b20 1388699 dzgjlc 74932 r152164 nb acusk n35f8rg8wxk gnamer helpme 2013 legacy cmrtazqlticrn1yx9lvkdnvtrhv3 nnnfaoqa0fcb rv umbra gl20002e cuyllc cjdz 5239 zhelatin 111379 airsoftware 468 311 s16628 abuse 039 etap genericrxep x97m faldesc 74192 1662 1388669 d1d303b9 qr 0052336b1 xuschz6yclkii0 bcayn 855522 s17514 00528e2c1 fago on 6763601 64720 8414 5267459 s3955638 dormsz lhnv afa6 perion bbsw epa 888a084b 60753 fex 0040f7f11 1523649 somoto 189645 d670e48 fmyj wnme dcdesq kutf 582656 d1d31080 bba smja rnd1gen qqpsw 34196 dzutsw a1a3 tn68 cmrtazom1bdhpbmqccjw3u5cvqa5 adbundle 36937 ddwqte liage d26a1552 titirez 00533b581 lmsy 74239 otynodzgej0 tiga fpzu b43c 48x6oc 5414 r3751 s19822481 jxdt l4uj pcclient2 trlj 0462 kpf dgzlogwfq w2skqwg fadok 99e1 optimum 0051ad261 falcomp msilzilla emdup 32295 17464 6529 d36397 0d43 t2v8zl9w wdext 26685368 54797 r11470 1033532 004b51ac1 fgo hawkeye naa1 asmalwns drstwex 7xhkt9 2b6e cmrtazp9 qudfwqrqif4o9jj7hy9 bancos bypassuac bc3 ajgr 251 288356 s4243210 33216 yz bzru nqpbmsbnsnk de7c8 34942 nettool c44d850d r238541 2c4445cd pe64 ped 7000000b1 imy 79c738ec minggy genericrxdw eixaacaxzhl ed9af4f5 33808 00529ce61 265216 vszh908hqb8 rt byh gof 7001009 004b69ce1 d21026 dotnetinject mdt tsuploader 13678 1037180 webalt 24926 ti ms03 s8 31647 005239691 e6d b1e 1lq7yb cmrtazqvwtbn6a4y0p nany87ars 40650649 005464661 smjq argm g14 gpvf r40064 8311298 xaw ms 969 84228 cqotzf 2370 tjnspy dc86b 9802373 22952562 qiwmonk funshion 7d8bdff2 moonlight ns1 winword 524706 jfc reposfxg 30615732 dgzlogxlaufwvj57sw 005274021 b4f7 004dabf41 355094 hd200132 ls8y 3d2xyze4 at3 ajz ajkr comame 9935 wbb 005380ab1 157927 175817 133200 41073223 buz nhz 242856 2509 233947 agent4 75cb ksr6mowueea 197330 fign 0f7e worry ipeb smjb 6912929 parasitic s1533829 bsn 444c29d5 euw 0053e3471 formbook genericrxfq 004c21251 xl d4 50e1 1f9 hfsautoa 004b9f111 436838d6 1762 40a stchinchara 6e8d4b4d mmnfaoqa0fcb d6517 2b4f aao diw ix 6752358 cylk dix 134881 mutopy chydo injectsection3dthreadc90a atf vboverlayd 0y9xtxsbb genericrxdz s148590 tp20 pxogtfckp9a kbdmai trojanaitinject nxp l6p7 373 refroso bkm s2722105 137 keyiso dostoxaltb remnants 226 d70578 ogebsad fiqprf d6b4 212992 9973 431220 73166 nobady 005223731 knnfaoqa0fcb csdimonetize 0049d1851 cred 576d oowcwixrc 9485 22330 crypgenasom dgzlogwkyi lv9zo9g 9965 00243f401 fqvk genericrxbj 00563cb01 cymx jklmno h81 700 k81 58451 scgeneric 1222 rowq d5d1a 520 90a smbh abv dm0 dc86a uz f6c malcol c3e1 c568 7648 idlekms 34128 s1645906 c4d6 423b 0053e8551 r240991 5211 malware00 glx d121ce xer server wakme d1e53f43 fqzc ybg 02012021 164 frx fd220077 cmrtazrwdw7ho5gvhfx0g1uba7k5 dllinject 7gephu nvr awdh2njzaii 77de9f82 egy 1016911 abc 260309 143210 31342921 macoute 30864105 282420 63793 fhod 64017 nwa 16460 fa21b634 s19010935 crab nptj 45640 63932 a06 gvm 2400 xq daj ex ajfk cmrtazqhj8ren3szdycsoouhqwjx hc010028 122880 dlf pupxfr d83aae9 proxy2 749 229376 004093e91 atk baj 1133 ebqrgw r061c0diq18 0025f28c1 s18975 s4124473 lordpe 376455 sh b86710e1 crypnan ga250444 gk210019 agentcrt 6444821 fjeunv 55640 00361abb1 109 nbj smartinstaller pullupdate 58574 eixaagfdxci 131842 om0 cwhe bnnfaoqa0fcb 579 aeda 059 pdjvgykj61m ldmon xsexwb5kcaa ddostool qvm00 kyqv 0051ba921 vbgeneric bqocvs hdern bsf ehppe smbx 9827011 casino amth sly fourshared fbqy 546 cmrtazr5rwdnh2zqgoqypxsw liu 1124542 bhn c629 10626 mmvi pqv firser sniffer aqra 003b1b581 cmrtazrpbo1cc hpzmukx0v4ygnx 085 8443 fxms c2f3b87f com 13838 d47ab8e6 95d31726 bodegun downldr2 mydoomlb fxgv b5b6 a0d4 afj c4903267 17188 vuhyo 40 gcq abdc bopofk ayu kryptikcrtd 614400 nw 946 04c543651 phdet 5982 1678 02012020 wb mbw 151 5s4u0d ptoy wswuhb2cnte zz6 koceg 066 34684 005338861 isda 233271 aqr trn 00545d801 fovernd 81408 wpepro 004f2a9d1 2338 sdk blackhole xxd 0aa6802d 241468 kg 0054e0831 7518 22060734 0000a49a1 morto 38609 383947 wews87 bestafera 508 fnnfaoqa0fcb 86253 rsm gbot 5i3ljh pwsteal ana 104958 r188849 navipromo 33464 coinbitminer 20353 17132 3738a hllo frp 7w15u1 exmgkf endom cssniv cvi mewsspyttc swb 1f7v8w3 1584 cmrtazojlrepwzug8xvlgulc4wqq qt 1cfa none generic13 pepm bewm gjjv 40618512 xolxo 294512 ok 5jedjj 135 gavir f1c1a346 260358 nm0 chu sm02 40416049 192198 goznym 40714676 gcfm jigsaw 6323519 9974 d4c 289 dsvtif 3514 pupxdr 1517 571 lot hpfareit ehcc anp 9881 00386dc51 1238 cmrtazoro3p pkk1ljano96s0 genericrxeb d5jcoq nadq 1a76837c ajj 316 d17243f5 abmi 381c35 ibt mnvo r87613 badjoke r011c0peb18 ncq d485eb bhh 6726549 1df62dfd 1311868 bankerspy fner fuc gace kranet 803365 pushdo b271 o81 onnfaoqa0fcb gjaj 4pn6tf 0040fa811 1388589 jg 2372240 swpatch bprotector tnnfaoqa0fcb 98304 96289 d2699df0 004e16831 cffd94f6 ezavgx zm 6840460 200006 ceb v3 9975 uwamson 178 ljdn 24507 ewrbpb aib genericrxda 31585 0da4 ctjizx 5610 290816 142 5f58ee87 16d v001 cmrtazq8hm8t6e6nnjsvbmbqwize fcl 3299862 le 0048f6021 sasquor buo 321 2f778cd31d os aacd lpi7 otfrem 4wlbfl 22548 tpct vuz 003437341 e9e txk malpacked3 nspm birele 7wuuah detplock 4372 adm01 broskod trickbot 5619 a9385d94 102399 bankerx piz 109208 192202 zona amnfaoqa0fcb 6mnfaoqa0fcb jamg 278037 tnem cqxk 6ev07a9ne5y gen20 aaa r223920 11216 s1439 r153600 feu a6a lbcw ecgyzw s4 qc77 nfcwbg 30fd658313 tonmye fjq bsoy qqrob l3eufaqmtpq generic01 cmrtazrx2lensnyjdrnoxx0ejkav trace rontkbr gaz ayh 32800 tn6a 004546b61 ae9c 0b5 00001b761 236c32ef aao8k8m5ene fake kgl 3jbhusbgsgc 1002245 155648 v4 vtfloader 236a fi08029a fmrv cdesc luev 00513dbd1 312396 9931 r251967 dc13894bea59 3bda332e acli skg7hmu f10001f11 449611 spambot 317 otorunp ae81 lgz 4665 0040f2f81 cmrtazozaifhmgyuzx1mbupltnsb tyk 1949202 40329 ktse 24461 ikis 254 df4dae5e hfsreno d21ce genericrxfk 52vn2u 75264 003eb2a51 dlz a8f0 9988 004f53391 ircbotpmf 2c85 efyboj 132q90f dc996 89442d75 1569280 992 9a19 d101d1 a9af s3402724 aumx nrww 143 cysfdn genericrxch gl26002b 2ff zmnfaoqa0fcb va ainslotgen 25572 aabb ve tn6r 1386343 apb 217 ayhi dtcontx cc1 bmx odixinaaf djh genericrxdm 00528a341 lua bitd mwal 40464789 jimmy foax moarider 28491 1123676 fkzdocha 1103301 znz 15cg12 xa chx 236510 461512 37577 676473 101673 ac1 239468 sap 294261 000172de1 3ff adc etsj pondre 00529cfa1 malware5 d57023 fa1 d54a08 cb8 gimemo exeaggmg 218 bk08455d mk det r3oogv1g6rm lpmg webpick b72 0053e00f1 004f87f01 cfoo 12451 d154dc7 azgw 138 to7p akkgg tipv 64158 trjndwnlder takuburan 36126 vmz b34d 335360 xes eyxnse anq 111 fok ahbp d26d758c ant 0051b9181 00526bf61 autoitdropper 1039007 6878631 24524 1683057 python 7002 php a7ec 7136 52eac414 pq3 geral 800 95706 hmnfaoqa0fcb bkyz 99df d20d14 tpoe ts ntrootkit bauh 173 msilamer bacteraloh 9992 004d9a1a1 zwr demmsd eaqemx coantor hhv eaeqpo lnnfaoqa0fcb 1867 7ohgyi 7f b420 lbzvun3aba0 r220107 0053eec71 307 tspy64 avoe nilage r39186 ae59 azno euqtlz poly glb fhb face 5047 njc qg bsw eee rpr r212554 adf dxfc tn6s ae69 6966 184 ab3b loc3 xejim 3322 0040f8ce1 fd250377 bouem f4aa37eb ee9x8nzaddk e83a6442 adclean 54297 0053c2731 pswtool 40760472 tav gamup f607e0e6 1mnfaoqa0fcb vdld dif qmz qhidpkoe 009dbfe7 dqrv crysis cmrtazozmzu7vyrt ftms9g5q8xi 8e8a1e4c fpuo aylpxx7fj fkkzve pue gen6 dzan tp87 1572864 bll srg bbindi loggerf1nd lm21 vmpbad r217426 qgz 0043eb071 1d5 64733 dkvt b348 j2 9986 38511e5d be9f59e1 gsj 59ticr 6931301 218120 uv 4b vbcheman c567 gz0c9kqr9uk ranky esgroc 9286313 s1601949 004a955a1 defaulttab 564590 r222991 a19d 1300677 26917654 fnrisw 6260335 40446742 938284 78624 14057 avemaria ade stone 283100 opkoelc c47526 opesup 900892 smalldl vittaliaent rmnfaoqa0fcb 0040f96e1 432de27a aaed 9961 cmrtazqqemefmiulzqkhllpq7ihy d1d9dfbb adplugin td8hmla1xe 34122 238 ckti jjb futurax 1432 edp cpssu vtfodtvm hwab 173464 hacktoolkms rfrahnb r33547 mdcd ni2uo2kkkji dgzlogpqi1twngpunq 78559 hottrend 12535 afea rved b943 downld agq 93251 amrr fnzs 4715 d2681cb3 kolovorot evacni fzmh 40605901 blackenergy awz 275309 malwarescope 6399 ath sisbot eh110001 brontoktiwihv edj 726 zb shade r251417 cjz 1fd9e307 pmnfaoqa0fcb 31060266 cloudatlas dllkitster aghnbqli 58237 oq3 aka 9980 64959 22793 6242658 30974842 75cdf7b3 534297 coinstealer 0053082d1 wdguj 2990 de89148 1343 lebag 00536d121 krotche mdj4 118784 s3232693 iminent pwsonlinegames generic36 malautoit onesyscare 197658 obfusransom aeffa 115527 9990 ospn kcl aeu aoa vpm 31464 smej5 namnfgc pay per 10b07bc7 d110b6 c4e2 atdc hai smic 225280 bbc af4a autovirus oneclick 005110401 9786317 r214462 dccd5 angn 0ed2 6688770 d14b27 ua 00001b701 048416 6726655 bm0 gf06020a pjau4te3gzk 2500 3dzo ywzn14 754 czli f1c645b9 boaxxe 6913203 unp jmnfaoqa0fcb networkworm centrumdownloader 134860 130 ctcm b351 ewbzkb bd01f5bc 143360 ptedmjd gy startpage2 r266571 aqf cmrtazqjwgkwm3v3nxrdbvt9hod2 2145 dake 226452 6717397 ba86 axkd hcjxpze bes xar h8oaepsa runner d1d2f360 bedep ae6e vxfyv faai dbddjv d2684d72 cgx c7c49bf4 r191213 533 nhyt m69i nskh 510 dfg 163840 fipp fih incredimail 54954240 prorat vij um mysr 74gte2 360 fbpa g81 e7740bd5 d2701745 mokes shelma hideproc ibu greb emh p0 skr smshoax esprot 589164 301 xwbc 2801 eyzqry 10883 51648 csrqtx cadsn downloadius ab8e 4yzjxl cmrtazog2tcoe48h1omyzanrpmdk mvaa 0050c0801 hc140064 04c4f6e11 airinstall dothetuk 23981 lgxv ae04 fynlos c689920 00558d391 r34837 d26a7e8d 0qw0 yv22xfexp2c ll 5876 bcagho 004f34121 41dee0 2973 f608 d267da37 0040f0f51 a460 7zle8v cmrtazok04s3inwc9hbyjvtrmxcx 03f 35784 grgy figixi corewarrior fragtor yh9tizao4 12815 dzvg r029c0odu18 mutabaha d1d653a3 sasser 5101616 30855801 c55b lvup wky ctblocker a7271ce8 cmrtazqeffgnrgma7pxungtbq1er 004c861e1 ucpeo masy 30681149 004f3e551 2591 0049a60c1 005235841 22014 ltd poweliks 7009 3542 glz naffy detrahere 28596 mimikatz xabo tn4u d1d33f50 1e235348 r223201 4007 vskal 9987 6048 jushed cmrtazpivwbisnm3zxbl21ck6j0u 5510 lbe dgzlogi1rwe0vaivgq agobot 212 40795793 d138fdfc trjndwnldr sm23 hapn codecpack genericrxcc ewkswi sfivrnqnen0 2153 36621 6734391 2427 gc1600bc a8bf 72249 to7o installtoolbar mau bvs 90112 fzo eliuzg xice 5093 radar03 868629 buzgq c9f45943 139264 11264 37b sz3 130577 6290448 1105402 tls lgnr 005325961 v2 djcr alr 13b iqnql6zs3w0 d56f7 aoo 139776 sfmhx 822a87a6 9o 6a70 qoty 285498 ug rre bgwk adi afm 0051a6801 fynx cabren kvmh008 uifcwsdhqaw fmw aabc ate deau 31313826 vnnfaoqa0fcb dr3 salpack ubar 2494724 40457812 294 21754 d26a2053 7v8oon 4749054 spammer 0053d33b1 avf amsq fxhgqawk genericrxcz 7x13gz s2039290 xfu lrb rhg bpvb deharo vgfz 11431 9r2ssq6eba4 hbj qp dgzlogl8hu6sq pfzw cqkuot cyr qv4 nbu dwcl 57619 dnw 005328801 9958 003c2aa61 suloc 62076 r169912 ymvyzoi774 browsefoxgen secrisk 293 34186 0040fa341 1115206 355 2981 0055e3fc1 lse3 d1d8ee91 fondu ek6 9842321 komodia r233959 genericrxgu auhbmjg 7c5 hnnfaoqa0fcb smhe r70572 29592951 crm 812zm8 3678 684408 fdukaz 19192 dlder r00gc0pho17 pgy aru efec4c38 ann 001098b91 6656 skybag tp0s edqrfj 206f 312 c2756321 004fb5821 ioj ckpw gs3 smas bero jnnfaoqa0fcb gardih 1f5 846 10300 mjgp ga310e71 134073 guagua sm7 ggxm 004aec821 1029502 279808 31252 ilcrypt 49e92c5f 107719 dsxo 87217 240482 mnnfaoqa0fcb lm0 342281 veil cf7251ac ayvv ovsm sfp 9cc1 61a1 79514 kmsactivation 2975 oeb jrahrfa 0055e3991 7ai7rr sm34 0040fa661 dpah escyac aja btu 6295758 acr sj en21f f0ns skill c1783266 axt genericrxek cil zlz cmrtazpzbzhwljuj9snfscg akwj lbub cqkxva ltxd skypee 7ac5 prepscramri 7vd4ia d766d37 107271 881 kasidet adtk d38287 61678 bok dinwood c2698446 gep rimecud m0xc oo genericrxbt nm1 pn 48641 eodtwh quasar ucfbtdjopfo 245760 zexaco imnfaoqa0fcb linm r217234 skynet 4kvwk0 afg 00e dgzlogounboubftuyw 290327 d1143 upack danginex wf 1134782 hxgb nimnulhqcb dkndtn dec001371 drhe 6222447 c1032537 x81 aec utilities ajcf tepely malkryp oooapset bam lydra 0052a94f1 us amakvoli ffc2199d gc16008b d73dad9 448747 sybddld bmnfaoqa0fcb phishbank 83057 268899 35607471 java 309691 ckxw m2kh bei s17644121 1036656 vercuser androidos cospet een aig sid helloworld 2b0e 318 9c59 306 kip d1bd6f9 r69813 ae3d 40158 0008c4a21 yhy cgeneric p5iau4rdbhw quireap nsismultidropper erbccv domniq r138132 5a57d81618 driverpacksu 0043c2cb1 dloade rf4 mbf 111aedf6 adwaretsklnk malwares gje 35 ebhf 437 da296 6623041 backdoor2 c1563124 xdvrogfa5mk smus8 dgzlogva91imnhjf8g prepscrampmf pd 1121817 0029f2001 r060c0dkn18 uzc 122 0004581b1 7850 aem 41005315 5a1d generickdcrtd s23076 31039 egdg s1386997 ad40 bomber diliman 296092 51774 dce 0040f8071 30609057 181360 8137f9 001dfdd71 7axq6t asb erakhl 0040f4ef1 gc130008 r116134 pk2i4espaga fmlkls fvwk ys3 bayrob 0040f7ba1 1e05 danabot 147673 fm0 21370247 afo bwc 6e82416d zs3 c468f38e pwsime aakr dyljez eaqekt 9970 hx 5059 81lqy7 2461429 bqgp e52c99 lj dgzlogv7ykqunnqspw y7 cvcprm 83630 35932339 378539 4z0n54 nbr qsyf r12280 qmh awfi xub llv9 r243005 c2918197 j84tk echangneti3 axbj da3f13 r220488 aba vv4 b23 wonton s4238554 agent5 rnk aoub oax bv4 43435 188416 c7ab 13108 d1e408ce r213194 xdb b51f zk 75a5 eigril adl9kyg 1081 fnkz genericrxes 34 87654 6612221a d12206 4682 10231 ff85 akda 004eadfb1 acp djpz 6483 shmfxw 000fa6611 sk fxp ue erasub 57122 babylon 004ee7aa1 6750 generickds 692 esaarl 43b28e91 36445891 fdk 87061 6503 43513227 toit 0048d3f71 324634 cheval 294519 ejyqci pushbot bse shadesrat jokeprogram bec miners 67775 366080 cmrtazo9j5lr w73w z2risli9ud 534sds 185 dealplay 0040f5921 rg genericrxps 71339 duciqm 3b21 vf pgz r223334 cvbc 7iu8j0 km0 silcon wj 0040fa391 spyskype uo0h3leuft4 nbk ri 2031661 rd2 3mz ngz wr3 bdm 00512af51 9979 7409200 0052908c1 tn6d s7 30972 b459 004e13371 genericrxgj 350975 berer0px6lo kzaj 4e45 chipdigita1 ddaa83 hsxs 8893 c301196 zienimj9ck4 afv d1da0267 agz convagent 0053f0d11 d4ee0115 002a1f9b1 ipb 1388690 005257651 sml2 0040f8431 004bb4d31 6299 63263 op 1103297 000404591 d5eca4 pupxaq 00050a041 jmb palibu 3072 305 bxdp 0053f9621 30618390 1838094 5mnfaoqa0fcb 272 111934 271148 hr3 004f50331 22891081 23261 20480 clipbanker l61qieaoyby c2822672 sramota ek2501a1 651d bhnz 1123929 awyg mew smfe c170119 fsus 6539596 smsbomber ws3 854 104 cmrtazoao4x v36grecgq100m2ze 11042 9035 pam 1e5 alien nhxj 6823 ongamez1qpb 1532 tnre 004c35b01 faw 1103345 79556412 zhda2mdrzqs 77e5fd77 teerac gx 105102 003c0f311 sr3 pdfcrypt 6726555 9e3 s3209673 17810 blaster cmrtazrrqzruuof8qpqqkfzg4 fgq 006376 s619495 9966 r049c0peb18 puamson vak mpxz hq d3a0c2 107 avj 5htsrg d169f777 smal lqh9 42c3e4 d35c86 9b8c 3761 bs5 waca hnsr 32373 eixaao3fyqn 1029 982 146985 2313 genericrxfu zbotpatched dtleju axespec 3mnfaoqa0fcb smb1 9a0e6078 454806 hktl gobot s19654475 genericrxcd nek jft 302178 rozenaa 293013 exploder azyb fcef 6a4 ouvlk 0fee 5c64 s4750685 dgzlogwxjsy49az lv m1z4 gen12 0000000c1 genericrxgb guotoolbart stormattackgen 9582 d21025 mphage agenttesla b332 2la d679 sw skc hpnoancooe dsqr m4vr r92523 5722 2168 xploit sbg 230 beqguj vzxe s2040978 dropperzbots fwc a7832c08 3771246 softobase xfp r240840 dqba ouqm 11ftgy3vaf0 agentgen 1860 mrr 63175 6723903 2mnfaoqa0fcb aekl 34654 56751122 inboxtoolbar skype daa0d9f716 40492610 equo bk2201c4 7290 r002c0pac19 xeb 0055219a1 005435291 fix62crxo3e fksz 72f8 312102 haperlock amy d26576dc 004f94a61 aucw belvima 80t7o6 r86174 tjtroj hllpphilis d70f3 nay gl290013 hwl 1845 ahpb 27664 fhifce r67182 a7f4 0052ea4e1 xs3 bmw eaob ha19001c fnz 058985 cgbl xnnfaoqa0fcb siz tig 740d4823 31702870 hyzbaukwkdpb 275138 etfj dbw 9969 ost dne cmrtazqhvbbqtzksu9yjnz3yk3k9 limpopo 245140 deagezc 7fyqfc xj documentcrypt b56e ata svchostxi cmrtazps71n8n8kg8ln7xovjfjyy c6895063 6726939 17330 70imchinese r158089 a60d 410649 30609377 s1384038 152 lu domainiq hwsbepsa ac47 oi 95232 1103313 fkme loy0 igz 201 ixrrou a7gw ilmp 6533 0uhzuxspwdi d4c3 click3 d147a2 553 downloader14 gxd afe3 ur3 00524cc21 34298 dra 39727 770019 yobrowser fusioncoredownldr 1940263 yek 58289 1774 all malpe ry cinmus apm 124928 ms3 speedbit 325120 r207864 192 hllip 17920 miner3 837 2lp5q i2 salrenmetie 2497 d4e1 mazo 2524 004ca2e41 bm1 lodbak autoruns amzz ff210038 i81 czzbaukwkdpb 15451 180311 smz3 relevantknowledge be74edcd cuxo fmnfaoqa0fcb 318829 00504e3f1 532839 pbot drom wum 7mjfx1ertvd 175908 ecilko d1de56a4 evasion r7739 270 tnjf cmrtazp9qaylxtqqn0frdxeeuhka s3293683 ba3d cryptd 6851 1913 s4231859 magniber ad21 341 cee mhnr 6872 856215 anmy aytz genericrxcx ek040362 emelent 004c2cea1 6108469 8r4 7fs21j malvertising bog alustind 003eb2561 mmz r226234 0052ef101 cryptdomaiq 4p0xzq obix preload disbi swp aubd6nni mq ad3710b8 d1d2e52d 227937 vhu asmalwrg f6 kolabck gen11 r159631 drbr rms r61194 hlk 004a54ec1 youxun ajk ays r232581 delpdldr 0049fde21 s5262759 fraudtool 6726648 165e8 gmfb gist jhz r3 427486 000c130800 bcnt 5690 b565 qnnfaoqa0fcb 00003f371 pqt botx genericrxbp uw j81 75059 56yhj8 335448 driverpro xorist dmfkjc 0301a1e1 6776766 ljke 004eb2461 914311 qcvui b1ea 004e48c71 auw 004c36c21 b275b5c8 l81 nisloder 31804174 150227 17390 38019 6822 32504 xyk 68032 30994007 bos1dvi 279528 agentattc emg r633 waltrix alb 9908 tnqb sumom 0053afa71 s295525 duivng nau enestedel 477191 0000020760 cxblji r178729 dxqnytxnf9i cmrtazqr4ojqlbpzvrlq3xd jk7h r235632 cryptowall3 1172092 tordev bjrkpr d1b2f xab b2cd vi uo gfu l4j9 bak yt tntc 6507 4678 lapka pqq 304 etgnjc 58880 q3 blo 7cxuln amiyxeni fff aks m7ec gamepass atgy ae0c oxkq aple 057677 obfuscate pluton z3auuv6x2ni 61e27dc3 cmrtazqi6uqcwmc1svk3 rnfkv 78235 303104 btvwx 759296 434532 ec083e89 6725383 bann c42d nnnn dynacur 6939 cd14 generickdv 37d2ec51 abjq 75090 d8d49 bbo wnlrw fe090027 3898 40607098 d2 igx 0054d1101 propagate 156795 idlebuddy 22411543 234779 bte 317822 ixdrffdqa dpgo 55672 577638 hzg6p71kxm 151552 bnq 6804648 qipz d116e0 smdm 56544 dae6 fjtpxk r242101 emu d26ea4b7 1130496 19838 gulf 0525 ircbrute 0483967f atbot smar5 rj nv suspici qvm27 pe1 d21ddc positive finds 17697 qsk 1656 hgiasoca evhziv nsu blgk ins cuqrlk crylock drlc 000280871 7b01 bsuq 005317961 bvp tru 7ghrh3 004bf1811 00193f571 hgiasoka 271 005402e41 cvh 7104546 epxkni r200894 bobic 26977 65a3 ekofiv 00528a331 0049a3451 1009062 fasg genericrxdb d85ed0 0027d35c1 146 139048 2fbbd27d pupxdl ou coolmirage ghq ctsnpxq one esgrxh 004f00e31 genkdz 380 8881 769 65186 0051170b1 s1850719 cqg0om jrw0 1222301 skl 196608 ija farrva pupdownloader cdsk 30d2953f agfs ab1c 004bf36f1 takc 0040f9001 4cc d47e 004d42671 309 005149bd1 edf 3278 fkh 0e09c2ed 2239 smvbri fowl 004dcd281 jras bamj pynamer screensaver 0052f5b51 7c9ea50f 292919 31852340 85441 42951 bang5mai 502 lrz 28e7301d bertle 7588 co1 c1722152 qd yaaagr7uydm bqw pupxfu yo b209 e7c9 0040f0591 d8c1032e virutas 4ocig0 92134 2wqyqhhm58c 1454 9910 d26a84ec cmrtazqkscsdo24jxujf0q aiebu bsc 77e8 amc ccu gaai49jk3xe 00524cc11 59675 huz bdf trfq ctkmgw kil zbu 10999 mt7t b43b 55698 asjf a32 s1507512 d634c3 d2684d6b winevar b971 c211827 d5c3b0640e antno 269 bxib 005257f41 r136046 tp7g 6782660 acl dgzlogy40ksmuvfppq r68544 dujpsg d3f91b kazg 0051bf5d1 1017878 s2322759 twmx krol 293800 smpo ahg yi genericrxdp eqwbbq genericrxbx 31086742 tn3v 02012421 0040f53f1 bicanotel dqxg cqkyjd icloaderpmf cis we 35f1dcef sc dbycya 7vf9i1 r238577 fb9a 20619344 ynnfaoqa0fcb 0051b4b91 mamianune tpn3 6222243 ab1 vlie 00574b221 elementsbrowser packy epicscale squarenet pl 34523691 asparnet recodrop 41003 c52b5045 deliric 115241 40872100 quervar mediaget 8hjrzn dom downloader16 lrse 225 36885 495277 04c5316a1 6a67 azu pupxge ema 1037947 wrw 21520 iez 004eb1381 e745fd msexcel hlubea wkee 4558 004d0f671 2551 125935 a144 aaqn mzmc 1825 riskgen r004c0dkn18 e365 bpdp nonwritableconta glmy cb27 0f2f waskyb 6169133 erausx smd2 0050d7171 bk083539 bjbc smq clnvwd generic35 ge2300be r194942 9880 2649 plocust 6086 gjue 7f64 fae 9950 ara smjc bzwen 54102 r19117 245080 7tcxjx amval agdg 771195 e5522e rokku 00507f2c1 s4058828 ifdx 41c79570 ckezxg aonb oy cf6c304d 24276 107235 wuz 9919 wq 2976 nezvfi 00531baf1 2961 a470 1332 shell 182 befd 3993 fmo bclt cmcgoi 10232 005475011 petya 786 004eb8b41 00540ed51 005187ae1 80973 1511 382 rk 005223751 85242 772221 004fce131 30618362 tapin 40507703 mofin cud 493539 003b505d1 40391479 ddpghf pwsx 175 systemoptimizer biy413e 1024 433 ayvx 615901 a24c obz 02212021 6968375 qadars 110 34214 akx 0020b3dd1 b1ed a585 aty 221922 292 fdla 322 1013604 zyx baw udsdangerousobject cmrtazqcrh1ryxiutj mm3dtyfc5 1d8hvxr5z3g s19755586 f8 overlayvm sogou 3943 afx uws odg 5929 lnkp cqikyo 8040 fdfi nevp 41560 453 cgjg d3c546 cin kee 116 ghkk fev 76f 005105151 d727 7953 girh cmrtazqmh4eyagbnoil6aiyht4e8 esouhq i3 fia 004f334d1 9536 740617 egvznv ewc r61679 azwe dsbot lje0 6912930 xaa d5df201 s1164383 dmx 134723 jusched acmc test tgz modred 005103801 9077 ek040404 30594223 d26a62e3 r252862 ek1601da c1262960 dumpex cox siscos genericrxgw bqv af4 d26855d1 genericrxcy 7mnfaoqa0fcb nav 431 zz7 2db89a8d fdt2h h12c4 cqkksr 29042 toggle 6657 wii ivx qxe arg hd200197 vidro 582 230440 aq3 liz innfaoqa0fcb 29364 cuxk mucc 3754 7847 ffx 9133 ivvdh wvd genericrxac bcl nal 34134 004b8ad71 2242 evqpzj elementa remoh u9u05uapnam af89 cpbmb ejwd b24f gamehuck r240933 rontokbroer hpju 54688 bzqcm ae54 f5d ethqlz amdp36bi 30005489 fd120164 a9 nhb foft ahxv r64039 313 s1322 vsn07f18 iz 2468 aecb jqdw d19bd4 xpaj 26112 777953 bruteforce fakeadobe elf crypt9 127 192738 9962 instmonetizer 5260 2dd4416f 00527e591 00501caa1 fntoeg bjs ozv xr aph 8b6469e8 s8319 vfdocz5fb8a ddc5 gxb bgu snare uxr bch 78kg5e5es0u cc81 pmo 004c3bbe1 3da5426b20 xr3 ob eicar pp 368266 cmrtazqo2emhqt27ry1hc6e3hfpu hxibepsa ecnvuh tnrh pwym uejo equationdrug s1738243 kz0o arr 29530 cxdc 6tos6b 0054f14d1 superthreat myd i12 72211 cmrtazqljltjjqs6a6hrwfvphrhj 7749 bda 28409 117444 t0 084 flo 74843 smp6 ruskillttc r254647 netpass hiddldprng 4a2fd3cb bd8302d 8b254866 74321 proxyagent smaly0 r95307 4902 ais r68917 wonfaoqa0fcb 95357 alx kgz 6bc6e652 8107 293402 873407 dq3 dnnfaoqa0fcb bhkg lr3 9508138 22760651 18817 110140 30886173 66980 kr3 gigex 394832 uzzbaukwkdpb 294324 dpsk 6258116 aoc abaq 3723264 wanacrypt d16876 kvm007 c1935883 smjs af45be ge230004 hgiasoqa bba1 cmx 119 c771 r252073 f54f 3833 gen35 ndie 15318 9939 rn kuy wv 1407811 abo 155087 6804558 34764 jwrzjc 1121814 1103317 l2llqvwijeu bnv 8528 backdr genericrxmj 49007 001f574c1 493428c6 cryptinno 42246 aohk genericrxhe un 022d e8c1 564138 scaranv xpyn 9f25 266240 b5b8 270334 e645c6 86016 awv aojtk8ki fnlses dfzab 7da1 amr a0lbm6ei pjfakxfdhlb ohz c161 xh 2756 genericrxgh 110592 xrw 2319 tble 1103304 wc8iungqxfq bandoo dil 004b93941 mhz r234778 dg3tlafxpgu da8a a0b3 d47815 mqw tj 85618 filetourinstaller ac4bv1ci 662 hie 34145 40403788 525583 js3 q4uvjvdiene 8159h4 005514aa1 27399 vitru adh 0313 hbg c121 220032 cmyr awv6n7hi b22c ponik 004b87ea1 1667584 deceptpcclean bw5iyagr9 200012 d9b5e1a91d debrisgen cebdbxd scrop 260330 3d01 5adb eqrggk 1243 529f1d16 bingoml 13224066 1364943 21317673 thz fbpf 449518 vucha 190938 1585 40879496 qt3 787141 87161 nrj ns3 113983 32593 30994 pvkw r115343 sjdzkyuweze 7t0lia b8 42919 auq 0055d0d21 nevwfb 0051abee1 esgrxe aulwr3f spigotpmf 9909 ejyj dtznyx b48d 6189 005048871 676 9941 c1897952 448218 cryect s19159836 7h11hc 004b90a21 gandiaom 005375751 6443152 msilfc 5943 alduui9imyk akan 80vzyz topis 5977aac7 48ubj9 004dd9411 6956954 244ee29 anv 004f69aa1 73c59cc5 offend c141 oscolef azbi r136020 2ae6 ptbbfld dpmgko 77543 c193876 yl 004b89ea1 6750707 120 z81 b0ca 6456 10b613c4 xbb offerinstaller gzx hgiasooa 0053e3481 20b26d e021 39ce5c2f 177 crkzmz 6840462 cmrtazr0awrsryqv4afe6dm5umo0 wm0 0040f8661 88706 abr 9766 fgr imj aykk d4c7e tpda brg hgiasoga 5085 tfn d53108b6 cwjb 2305 puz 00523fd21 tx r039c0cis18 fho dvz zstph qifa wg x3bsu6yu7db 1110309 28483 adj od9hmjddgab watermarkhqc 6717393 smg2 6847892 cztf 00557fc41 xihet lwnh dosb 7829 d14705 d733 gm0 li7l powershell 30685140 lxqm 289804 1114815 422 52490 blx 6622929 cryptodef r00ec0cgj18 0040f7411 atu hgzd qbu0as1tzla 852 bcxo 4e09 slimware 40461586 aoe r002c0cjm19 1g nmz dmlm 057726 ans badcert 30922015 ransim bqtxv cycbot a56e3671 ga25096a vka 005424571 178982 riz dosp 0052d6bc1 111624 539 ntq 307203 chifrax ac0 242405 a403 b313 winyahoo fakerean gatortrickler 3049 clfug 3e8a7 6765990 gsy 2310 8dyvchv0auc 34410 drf9a1xlw4u 7r9yms pwsbanker s714409 195 8390 dzod 18812 xap c701613 eqw 9723 144 1264770 wampori toolbarcrawler ctye genericrxay hpdridex aby 2473622 s1819456 193765 7u8vdc r247308 a338 5334 vnebt psb trojanaitminer lq um0 s1969356 dipkgc 19f4a4 tdongs mydoomdelf r8428 170 cqkxjc r38550 ghnl0ugiftc gc220002 qy acf111f belarus 385b 200087 cmt
'''

        self.encoder = Encoder(200, pct_bpe=0.88)
        self.encoder.fit(self.test_corpus.split('\n'))


    # def parse_scan_report(self, idx):

    #     # Find the file path that contains the target scan report
    #     line_path = None
    #     for file_path in self.line_paths:
    #         if idx - len(self.line_offsets[file_path]) < 0:
    #             line_path = file_path
    #             break
    #         idx -= len(self.line_offsets[file_path])

    #     # Seek to first byte of that scan report in file path
    #     start_byte = self.line_offsets[line_path][idx]

    #     # Read report from file
    #     with open(file_path, "r") as f:
    #         with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as f_mmap:
    #             f_mmap.seek(start_byte)
    #             line = f_mmap.readline()
    #             report = json.loads(line)["data"]["attributes"]
    #     md5 = report["md5"]
    #     sha1 = report["sha1"]
    #     sha256 = report["sha256"]
    #     scan_date = report["last_analysis_date"]
    #     scan_date = dt.fromtimestamp(scan_date).strftime("%Y-%m-%d")

    #     # Parse AVs and tokens from scan report
    #     av_tokens = {}
        
    #     for av in report["last_analysis_results"].keys():

    #         # Normalize name of AV
    #         scan_info = report["last_analysis_results"][av]
    #         av = AV_NORM.sub("", av).lower().strip()

    #         # Skip AVs that aren't supported
    #         if av not in self.supported_avs:
    #             continue

    #         # Use <BEN> special token for AVs that detected file as benign
    #         if scan_info.get("result") is None:
    #             tokens = [BEN]
    #         else:
    #             label = scan_info["result"]
    #             tokens = tokenize_label(label)[:self.max_tokens-2]
    #         av_tokens[av] = tokens

    #     return av_tokens, md5, sha1, sha256, scan_date

    def parse_scan_report(self, idx):

        # Find the file path that contains the target scan report
        line_path = None
        for file_path in self.line_paths:
            if idx - len(self.line_offsets[file_path]) < 0:
                line_path = file_path
                break
            idx -= len(self.line_offsets[file_path])

        # Seek to first byte of that scan report in file path
        start_byte = self.line_offsets[line_path][idx]

        # Read report from file
        with open(file_path, "r") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as f_mmap:
                f_mmap.seek(start_byte)
                line = f_mmap.readline()
                report = json.loads(line)
        md5 = report["md5"]
        sha1 = report["sha1"]
        sha256 = report["sha256"]
        scan_date = report["scan_date"]
        # scan_date = dt.fromtimestamp(scan_date).strftime("%Y-%m-%d")

        scan_date = dt.strptime(scan_date, "%Y-%m-%d %H:%M:%S")
        scan_date = scan_date.strftime("%Y-%m-%d")

        # Parse AVs and tokens from scan report
        av_tokens = {}
        
        for av in report["scans"].keys():

            # Normalize name of AV
            scan_info = report["scans"][av]
            av = AV_NORM.sub("", av).lower().strip()

            # Skip AVs that aren't supported
            if av not in self.supported_avs:
                continue

            # Use <BEN> special token for AVs that detected file as benign
            if scan_info.get("result") is None:
                tokens = [BEN]
            else:
                label = scan_info["result"]
                tokens = tokenize_label(label)[:self.max_tokens-2]
            av_tokens[av] = tokens

        return av_tokens, md5, sha1, sha256, scan_date


    # def tok_to_tensor(self, tok):
    #     """Return a tensor representing each char in a token"""
    #     if tok in self.special_tokens_set:
    #         tok = [SOW, tok, EOW]
    #     else:
    #         tok = tok[:self.max_chars-2]
    #         tok = [SOW] + [char for char in tok] + [EOW]
    #         print(tok)
    #     tok += [PAD]*(self.max_chars-len(tok))
    #     print([self.alphabet_rev[char] for char in tok])
        # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        # return X_tok


    # BPE IMPLEMENTATION
    # def tok_to_tensor(self, tok):
    #     """Return a tensor representing each char in a token"""

    #     if tok in self.special_tokens_set:
    #         tok = [SOW, tok, EOW]
    #     else:
    #         tok = tok[:self.max_chars-2]
    #         tok = [SOW] + self.encoder.tokenize(tok)[1:-1] + [EOW]
    #         # print(tok)
    #     tok += [PAD]*(self.max_chars-len(tok))
    #     print(tok)
    #     # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
    #     X_tok = torch.LongTensor(next(self.encoder.transform(tok)))
    #     # print(next(encoder.transform(tok)))
    #     # exit(0)
    #     # print("X_TOKEN:")
    #     print(X_tok.shape)
    #     # print(X_tok)
    #     return X_tok

    #NEW BPE IMPLEMENTATION TO MEET WORD DOCUMENT
    # def tok_to_tensor(self, tok):
    #     """Return a tensor representing each char in a token"""

    #     if tok in self.special_tokens_set:
    #         tok = [tok]
    #     else:
    #         #why am i substracting 2 here?
    #         tok = tok[:self.max_chars-2]
    #         tok = [] + self.encoder.tokenize(tok)[1:-1]
            
    #     tok += [PAD]*(self.max_chars-len(tok))
    #     # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
    #     print(tok)
    #     X_tok = torch.LongTensor(next(self.encoder.transform(tok)))
    #     #should convert ['ge', 'ne', 'ri', 'c'] into like [23, 53, 65, 21] (which is a longTensor)

    #     # print(next(encoder.transform(tok)))
    #     # exit(0)
    #     # print("X_TOKEN:")
    #     print(X_tok.shape)
    #     print(X_tok)
    #     return X_tok
    
    # i think im bpe-ing wrong:
    def tok_to_tensor(self, tok):
        """Return a tensor representing each char in a token"""

        if tok in self.special_tokens_set:
            tok = [tok]
            # print(tok)
            # sys.stdout.flush()

            num_rep_label = [self.alphabet_rev[char] for char in tok]
            # print(num_rep_label)
            # sys.stdout.flush()

            num_rep_label += [0]*(self.max_chars-len(num_rep_label))
            # print(num_rep_label)
            # sys.stdout.flush()

            return torch.LongTensor(num_rep_label)
        else:
            #why am i substracting 2 here?
            tok = [tok[:self.max_chars-2]]

        # tok += [PAD]*(self.max_chars-len(tok))
        # X_tok = torch.LongTensor([self.alphabet_rev[char] for char in tok])
        # print(tok)
        # sys.stdout.flush()

        tok_string = ""
        for i in range(len(tok)):
            tok_string += tok[i] + " "

        # print(tok_string)
        # sys.stdout.flush()
        num_rep_label = next(self.encoder.transform([tok_string]))
        # print(num_rep_label)
        # sys.stdout.flush()
        num_rep_label += [0] * (self.max_chars-len(num_rep_label))

        #THIS IS THE REASON WHY the "20" dimension exists!

        # print(num_rep_label)
        # sys.stdout.flush()

        X_tok = torch.LongTensor(num_rep_label)

        #should convert ['ge', 'ne', 'ri', 'c'] into like [23, 53, 65, 21] (which is a longTensor)

        # print(next(encoder.transform(tok)))
        # exit(0)
        # print("X_TOKEN:")
        # print(X_tok.shape)
        # print(X_tok)
        return X_tok
    
    # new implementation of bpe based on emailed word document
    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

        # AV_tokens looks like this: [[malware, win32, xyz], [trojan, linux, abc], [benign, win32, def]]

        # Construct X_scan from scan report
        X_scan = []
        for av in self.avs:
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:

                token_sentence = []
                # BPE encoded tokens:
                for token in av_tokens[av]:
                    token_sentence += token + " "

                tokenized_sentence = self.encoder.tokenize(token_sentence)

                Xi = ["<SOS_{}>".format(av)] + tokenized_sentence + [EOS]

                print(Xi)
                sys.stdout.flush()

            Xi += [PAD]*(self.max_tokens-len(Xi))
            X_scan += Xi

        # Convert X_scan to tensor of characters
        X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan:
            # X_scan_char = np.concatenate((X_scan_char, torch.LongTensor(next(self.encoder.transform(tok))).reshape(1, -1)))
            X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

        # print(X_scan_char.shape)
        # print(X_scan_char)
        # sys.stdout.flush()

        # Construct X_av from list of AVs in report
        X_scan = torch.as_tensor(X_scan_char)
        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        return X_scan, X_av, md5, sha1, sha256, scan_date
    
    # def __getitem__(self, idx):

    #     # Parse scan report
    #     av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

    #     # AV_tokens looks like this: [[malware, win32, xyz], [trojan, linux, abc], [benign, win32, def]]

    #     # Construct X_scan from scan report
    #     X_scan = []
    #     for av in self.avs:
    #         if av_tokens.get(av) is None:
    #             Xi = ["<SOS_{}>".format(av), ABS, EOS]
    #         else:

    #             bpe_tokenized_sentence = []
    #             # BPE encoded tokens:
    #             for token in av_tokens[av]:
    #                 bpe_tokenized_sentence += self.encoder.tokenize(token)[1:-1]

    #             Xi = ["<SOS_{}>".format(av)] + bpe_tokenized_sentence + [EOS]
    #         Xi += [PAD]*(self.max_tokens-len(Xi))
    #         X_scan += Xi

    #     # Convert X_scan to tensor of characters
    #     X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
    #     for tok in X_scan:
    #         X_scan_char = np.concatenate((X_scan_char, torch.LongTensor(next(self.encoder.transform(tok))).reshape(1, -1)))

    #     # print(X_scan_char.shape)
    #     # print(X_scan_char)
    #     # sys.stdout.flush()

    #     # Construct X_av from list of AVs in report
    #     X_scan = torch.as_tensor(X_scan_char)
    #     X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
    #     X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
    #     return X_scan, X_av, md5, sha1, sha256, scan_date


    #OLD CHARACTERBERT IMPLEMENTATION
    # def __getitem__(self, idx):

    #     # Parse scan report
    #     av_tokens, md5, sha1, sha256, scan_date = self.parse_scan_report(idx)

    #     # Construct X_scan from scan report
    #     X_scan = []
    #     for av in self.avs:
    #         if av_tokens.get(av) is None:
    #             Xi = ["<SOS_{}>".format(av), ABS, EOS]
    #         else:
    #             Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
    #         Xi += [PAD]*(self.max_tokens-len(Xi))
    #         X_scan += Xi

    #     # Convert X_scan to tensor of characters
    #     X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
    #     for tok in X_scan:
    #         X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

    #     print(X_scan_char.shape)
    #     print(X_scan_char)
    #     sys.stdout.flush()

    #     # Construct X_av from list of AVs in report
    #     X_scan = torch.as_tensor(X_scan_char)
    #     X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]
    #     X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
    #     return X_scan, X_av, md5, sha1, sha256, scan_date

    


    def __len__(self):
        return self.num_reports


class PretrainDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for pre-training."""
        super().__init__(data_dir, max_tokens=max_tokens)


    def __getitem__(self, idx):

        # Parse scan report
        av_tokens, md5, _, _, scan_date = self.parse_scan_report(idx)

        # Randomly select one AV to hold out (train only)
        # Construct Y_label from held-out AV's label
        Y_label = []
        Y_av = random.choice(list(av_tokens.keys()))
        Y_label = ["<SOS_{}>".format(Y_av)] + av_tokens[Y_av] + [EOS]
        Y_label += [PAD]*(self.max_tokens-len(Y_label))
        av_tokens[Y_av] = None

        # Randomly select 5% of tokens to be replaced with MASK
        Y_idxs = [0] * self.num_avs
        rand_nums = np.random.randint(0, 100, size=self.num_avs)
        pred_tokens = set()
        for i, (av, tokens) in enumerate(av_tokens.items()):
            if tokens is None:
                continue
            if rand_nums[i] < 5:
                token_idxs = [i+1 for i, tok in enumerate(tokens) if not
                              tok.startswith("<") and not tok.endswith(">")]
                if not len(token_idxs):
                    continue
                Y_idx = random.choice(token_idxs)
                Y_idxs[self.av_vocab[av]-1] = Y_idx
                pred_tokens.add(tokens[Y_idx-1])

        # Construct X_scan from scan report
        X_scan = []
        for av in self.avs:
            if av_tokens.get(av) is None:
                Xi = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi = ["<SOS_{}>".format(av)] + av_tokens[av] + [EOS]
            Xi += [PAD]*(self.max_tokens-len(Xi))
            X_scan += Xi

        # Construct X_av from list of AVs in report
        X_av = [av if av_tokens.get(av) is not None else NO_AV for av in self.avs]

        # Construct Y_scan from 5% of held-out tokens
        Y_scan = []
        for i, av in enumerate(self.avs):
            Y_idx = Y_idxs[i]
            if Y_idx > 0:
                Y_scan.append(X_scan[i*self.max_tokens+Y_idx])

        # MASK any tokens in pred_tokens 80% of the time
        # 10% of the time, replace with a random token
        # 10% of the time, leave the token alone
        rand_nums = np.random.randint(0, 100, size=self.num_avs*self.max_tokens)
        for i, tok in enumerate(X_scan):
            if tok in pred_tokens:
                if rand_nums[i] < 80:
                    X_scan[i] = MASK
                elif rand_nums[i] < 90:
                    X_scan[i] = self.token_vocab_rev[random.randint(5, self.vocab_size-1)]

        # Convert X_scan to tensor of characters
        X_scan_char = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan:
            X_scan_char = np.concatenate((X_scan_char, self.tok_to_tensor(tok).reshape(1, -1)))

        # Convert to LongTensor
        X_scan = torch.as_tensor(X_scan_char)
        X_av = torch.LongTensor([self.av_vocab[av] for av in X_av])
        Y_scan = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_scan])
        Y_idxs = torch.LongTensor(Y_idxs)
        Y_label = torch.LongTensor([self.token_vocab[tok] if self.token_vocab.get(tok) is not None else self.token_vocab[UNK] for tok in Y_label])
        Y_av = torch.LongTensor([self.av_vocab[Y_av]-1])

        return X_scan, X_av, Y_scan, Y_idxs, Y_label, Y_av, md5, scan_date


class FinetuneDataset(AVScanDataset):

    def __init__(self, data_dir, max_tokens):
        """AVScan2Vec dataset class for fine-tuning."""
        super().__init__(data_dir, max_tokens=max_tokens)

        # Load idxs of similar files
        similar_idx_path = os.path.join(data_dir, "similar_ids.pkl")
        with open(similar_idx_path, "rb") as f:
            similar_idxs = pickle.load(f)
        self.similar_idxs = {idx1: idx2 for idx1, idx2 in similar_idxs}
        self.num_reports = len(self.similar_idxs.keys())

    def __getitem__(self, idx):

        # Parse scan reports
        av_tokens_anc, md5, _, _, scan_date = self.parse_scan_report(idx)
        idx_pos = self.similar_idxs[idx]
        av_tokens_pos, md5_pos, _, _, _ = self.parse_scan_report(idx_pos)

        # Construct X_scan_anc and X_scan_pos
        X_scan_anc = []
        X_scan_pos = []
        for av in self.avs:
            if av_tokens_anc.get(av) is None:
                Xi_anc = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_anc = ["<SOS_{}>".format(av)] + av_tokens_anc[av] + [EOS]
            if av_tokens_pos.get(av) is None:
                Xi_pos = ["<SOS_{}>".format(av), ABS, EOS]
            else:
                Xi_pos = ["<SOS_{}>".format(av)] + av_tokens_pos[av] + [EOS]
            Xi_anc += [PAD]*(self.max_tokens-len(Xi_anc))
            X_scan_anc += Xi_anc
            Xi_pos += [PAD]*(self.max_tokens-len(Xi_pos))
            X_scan_pos += Xi_pos

        # Convert X_scan_anc and X_scan_pos to tensors of characters
        X_scan_char_anc = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        X_scan_char_pos = self.tok_to_tensor(CLS).reshape(1, -1) # (1, max_chars)
        for tok in X_scan_anc:
            X_scan_char_anc = np.concatenate((X_scan_char_anc, self.tok_to_tensor(tok).reshape(1, -1)))
        for tok in X_scan_pos:
            X_scan_char_pos = np.concatenate((X_scan_char_pos, self.tok_to_tensor(tok).reshape(1, -1)))

        # Construct X_av_anc, X_av_pos from lists of AVs in reports
        X_av_anc = [av if av_tokens_anc.get(av) is not None else NO_AV for av in self.avs]
        X_av_pos = [av if av_tokens_pos.get(av) is not None else NO_AV for av in self.avs]

        X_scan_anc = torch.as_tensor(X_scan_char_anc)
        X_av_anc = torch.LongTensor([self.av_vocab[av] for av in X_av_anc])
        X_scan_pos = torch.as_tensor(X_scan_char_pos)
        X_av_pos = torch.LongTensor([self.av_vocab[av] for av in X_av_pos])
        return X_scan_anc, X_av_anc, X_scan_pos, X_av_pos, md5, md5_pos, scan_date


    def __len__(self):
        return self.num_reports
