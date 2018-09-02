# -*- coding:utf-8 -*-
import os
import json
import math
import numpy as np
import pickle
import sys
# from multiprocessing import Pool
from numpy import dtype

jieba_train_path = '../data/jieba-data/data_train/'
jieba_valid_path = '../data/jieba-data/data_valid/'
jieba_test_path = '../data/jieba-data/data_test/'
thulac_train_path = '../data/thulac-data/data_train/'
thulac_valid_path = '../data/thulac-data/data_valid/'
thulac_test_path = '../data/thulac-data/data_test/'
sourcefile = '../cail_0518/cail2018_big.json'
filenamesplited = '../sources/splited_words_big.txt'
filenamecut = '../sources/cut_text_big.txt'
drop_indexlist = [301, 818, 1791, 2084, 4157, 4888, 5351, 6012, 6015, 6624, 9150, 9565, 10703, 12910, 13081, 15144, 15556, 15885, 17041, 18639, 19166, 21370, 25548, 26032, 29944, 30168, 31232, 31345, 31966, 32687, 33098, 40311, 41213, 41460, 42908, 42977, 43057, 44312, 46476, 47431, 49692, 50648, 56339, 56921, 57288, 57993, 60262, 60348, 60436, 61135, 63123, 63389, 64459, 66514, 67873, 72629, 72897, 75057, 75983, 76540, 79369, 79415, 82386, 83708, 85103, 86540, 88533, 90086, 91138, 91336, 91348, 91462, 92926, 93274, 94626, 99230, 100197, 101161, 102613, 103749, 104831, 104908, 105638, 108799, 108825, 110161, 114749, 114868, 115423, 118253, 120466, 121411, 121900, 122642, 122921, 123969, 124489, 126469, 126535, 127135, 128030, 129148, 129907, 131735, 134057, 134151, 136527, 138340, 139652, 142148, 143631, 146286, 146372, 147218, 147257, 147458, 148082, 149469, 149992, 152994, 155550, 155729, 160044, 160825, 165433, 166056, 166134, 167237, 168561, 168841, 168853, 169320, 170081, 173813, 180323, 180685, 183613, 184611, 186642, 187307, 188656, 188757, 190623, 192185, 193156, 194901, 196052, 201406, 201838, 203674, 204493, 205323, 205434, 206284, 207257, 209442, 209451, 209576, 211728, 213479, 214633, 214760, 215561, 220418, 224666, 225470, 226204, 227236, 227974, 228264, 229934, 230375, 230886, 232531, 232640, 239036, 239996, 243820, 245121, 245420, 251226, 253514, 254473, 255969, 258363, 259472, 262426, 265632, 269621, 269691, 272749, 276920, 278244, 278695, 279062, 279546, 280037, 283095, 283100, 285914, 286452, 286961, 287657, 287921, 288229, 289802, 291157, 293038, 293974, 298620, 299102, 301518, 302242, 303373, 303472, 305711, 306147, 307014, 307687, 307977, 309045, 310782, 310802, 311176, 311909, 313649, 314133, 314682, 314767, 315601, 316998, 319136, 320215, 323174, 324043, 326179, 327479, 327615, 328737, 328779, 331187, 333163, 333772, 334234, 334321, 334461, 335073, 335884, 336188, 337717, 338033, 338672, 339870, 340005, 340179, 347719, 349052, 350337, 350855, 353968, 354049, 355770, 358621, 359363, 360825, 361749, 363688, 364735, 366245, 367539, 367739, 368500, 369116, 370442, 372942, 374812, 376743, 376959, 379634, 379975, 380316, 382752, 386228, 387916, 393854, 395707, 396575, 398429, 398546, 399935, 399963, 400226, 401716, 403488, 404259, 406834, 406878, 407310, 407442, 407987, 409016, 409095, 410614, 411583, 413431, 417522, 417951, 418317, 418601, 418936, 418953, 419040, 420337, 423081, 424800, 425022, 425234, 426329, 429118, 429801, 430310, 431426, 433220, 433813, 439436, 440474, 443256, 443474, 445290, 448667, 450369, 450658, 450675, 451554, 451856, 452839, 455092, 456041, 457464, 457523, 458257, 458591, 459192, 459354, 459502, 459517, 460662, 460883, 462484, 463748, 464455, 466549, 468959, 471442, 471984, 474563, 475116, 476633, 479251, 479892, 479943, 482495, 483756, 484502, 485067, 489867, 490070, 490447, 490898, 492129, 492542, 497681, 497687, 498424, 499212, 501348, 501604, 502084, 502981, 503538, 510356, 510468, 512022, 513359, 517398, 517431, 519064, 519068, 520367, 523083, 525052, 527280, 528152, 528371, 529866, 530602, 531114, 532194, 532410, 534021, 534244, 535062, 535782, 538064, 539528, 540317, 540672, 541587, 541884, 542358, 542412, 543158, 548880, 550915, 552992, 556226, 557387, 559256, 561902, 563168, 563717, 564436, 565912, 568555, 572616, 572929, 574068, 574718, 575011, 576346, 580102, 581383, 581477, 582773, 584293, 585170, 586431, 586852, 587405, 589180, 590265, 591603, 592117, 593605, 593681, 597435, 598414, 598550, 598686, 600346, 600445, 602833, 602918, 605340, 606431, 610773, 611481, 612831, 613658, 613889, 614344, 614714, 615117, 617067, 617187, 617654, 617666, 618117, 620177, 620300, 622922, 623131, 624813, 626082, 626928, 626987, 627275, 627698, 627858, 627914, 628080, 628182, 628257, 628504, 629021, 630910, 631325, 632004, 632708, 633153, 634080, 635937, 638776, 640486, 642091, 647052, 648072, 648687, 648943, 648949, 649658, 650105, 651893, 661585, 661693, 661709, 661864, 662626, 663212, 664180, 665371, 666621, 668041, 668062, 669143, 669675, 672214, 672431, 674261, 674695, 674734, 675903, 676221, 677192, 678297, 679569, 680869, 683341, 683358, 683871, 684843, 685394, 686652, 688542, 688607, 690874, 691945, 692617, 692648, 692671, 694707, 694786, 694902, 695004, 697502, 701078, 701271, 701687, 702202, 706869, 707410, 707628, 708291, 710660, 712438, 713411, 713452, 713911, 715976, 716488, 716562, 717324, 717702, 718808, 720193, 722727, 722937, 725721, 726100, 726140, 729661, 730319, 732096, 733235, 734348, 734349, 734977, 736299, 743147, 743458, 746336, 746368, 747233, 748558, 748663, 749146, 749350, 749642, 749900, 751833, 752237, 752677, 753971, 755044, 761559, 762193, 762679, 765857, 768151, 769096, 769470, 769501, 769641, 770524, 772000, 772931, 773435, 773644, 773824, 773912, 777519, 778782, 779117, 779430, 780418, 781185, 781681, 783439, 785562, 785715, 790877, 791227, 794771, 796225, 796402, 797837, 803969, 804211, 805718, 806746, 806916, 807080, 807229, 808970, 809986, 811356, 811776, 813520, 818032, 818845, 822624, 824881, 824995, 827016, 827200, 828615, 828782, 829344, 830937, 833383, 834873, 837071, 838073, 839401, 839764, 839769, 840424, 841078, 843033, 845031, 846952, 850196, 853894, 854117, 854119, 854604, 855044, 855851, 857441, 858531, 860152, 861573, 863045, 867198, 867922, 868333, 870447, 871996, 872295, 874785, 875305, 876802, 877471, 878270, 878875, 879806, 882216, 883372, 886900, 890807, 891770, 892550, 893368, 896222, 900346, 900769, 902074, 902267, 904086, 904690, 906744, 907395, 907726, 908437, 908518, 908898, 911152, 914740, 916551, 917285, 917545, 918928, 920986, 922538, 922877, 923124, 924030, 924862, 928247, 928375, 929574, 930030, 930253, 932176, 933203, 935473, 936346, 937497, 939073, 939236, 939644, 940736, 943149, 945889, 946429, 950340, 950594, 952730, 953426, 954579, 957915, 958161, 959406, 959512, 960494, 962378, 964345, 965839, 967048, 967058, 968503, 968537, 969913, 970184, 971069, 973730, 973910, 974981, 975191, 976167, 976641, 977419, 980436, 981468, 982325, 982695, 982880, 983232, 984742, 984841, 984986, 986301, 986584, 986744, 988177, 988320, 988459, 992030, 995246, 995439, 997134, 999236, 1000753, 1001146, 1001697, 1002283, 1002872, 1006580, 1007982, 1008381, 1008865, 1015091, 1016861, 1019178, 1021706, 1022533, 1022988, 1026244, 1027938, 1029163, 1029194, 1030069, 1030215, 1030397, 1031727, 1033924, 1035698, 1038227, 1038633, 1039322, 1040669, 1040893, 1041451, 1043232, 1047547, 1049704, 1052291, 1053428, 1054081, 1055466, 1059724, 1059743, 1060505, 1060870, 1062717, 1063421, 1064602, 1066072, 1066966, 1069847, 1072009, 1074288, 1074815, 1075033, 1076270, 1077698, 1080735, 1082185, 1082963, 1084918, 1086739, 1089284, 1090770, 1091163, 1092280, 1092452, 1094240, 1094760, 1097674, 1099127, 1100644, 1102810, 1103304, 1105971, 1106385, 1109027, 1111175, 1111408, 1113073, 1113817, 1120599, 1120684, 1121356, 1121562, 1122663, 1124638, 1125152, 1125925, 1126580, 1128035, 1130878, 1131024, 1132150, 1132603, 1133000, 1133398, 1134756, 1135174, 1138333, 1139457, 1139954, 1140379, 1143241, 1143825, 1145335, 1147585, 1150476, 1150567, 1151301, 1152117, 1152403, 1154894, 1156835, 1158307, 1158437, 1160653, 1162072, 1163863, 1164410, 1165880, 1168551, 1170182, 1170382, 1171293, 1173705, 1173792, 1175196, 1175304, 1176504, 1176720, 1176931, 1177071, 1177105, 1177444, 1179241, 1180412, 1180867, 1182511, 1182521, 1183312, 1184570, 1185154, 1186172, 1187582, 1189173, 1189740, 1189892, 1190343, 1191015, 1191283, 1191319, 1191517, 1193675, 1196107, 1197947, 1199256, 1199693, 1201177, 1201307, 1201732, 1203941, 1204309, 1205779, 1206764, 1206961, 1208773, 1209040, 1212419, 1214485, 1214707, 1214958, 1218166, 1218844, 1218933, 1222609, 1223340, 1223761, 1224198, 1227619, 1228659, 1240573, 1240777, 1240987, 1242902, 1243507, 1243587, 1244196, 1244564, 1245215, 1247789, 1250196, 1250251, 1251035, 1252086, 1252869, 1253757, 1254484, 1256429, 1256620, 1258730, 1261513, 1262994, 1263766, 1265200, 1267166, 1270679, 1271809, 1273404, 1276122, 1276364, 1278903, 1280667, 1282511, 1283345, 1284307, 1286514, 1287308, 1291828, 1294521, 1297056, 1300602, 1300936, 1303362, 1309012, 1309769, 1310327, 1310522, 1311859, 1312225, 1312531, 1315252, 1315430, 1316044, 1316323, 1316655, 1317068, 1317342, 1318816, 1319426, 1319501, 1319926, 1321437, 1322729, 1323283, 1323768, 1324124, 1325002, 1325057, 1325540, 1325656, 1326649, 1327325, 1327383, 1328035, 1328536, 1328546, 1328720, 1332264, 1332798, 1332820, 1333322, 1334201, 1334234, 1336456, 1337304, 1338731, 1339417, 1343242, 1346607, 1347147, 1349027, 1350072, 1350143, 1351113, 1351919, 1352443, 1353603, 1353632, 1353956, 1355397, 1355494, 1356260, 1356375, 1356409, 1357504, 1358603, 1358938, 1359608, 1359861, 1360373, 1361153, 1364850, 1365607, 1366863, 1367545, 1368300, 1369245, 1369968, 1371531, 1372159, 1373348, 1373394, 1373535, 1373837, 1374731, 1375415, 1377107, 1378460, 1378732, 1380822, 1381302, 1381763, 1383933, 1384860, 1385792, 1386346, 1386612, 1386929, 1388118, 1388443, 1388590, 1389986, 1391585, 1392836, 1393502, 1394126, 1395649, 1395992, 1396013, 1396365, 1398161, 1398349, 1399477, 1399598, 1400904, 1402790, 1403740, 1403980, 1405537, 1405600, 1405941, 1407145, 1410032, 1410713, 1412820, 1417362, 1417428, 1418942, 1419379, 1420017, 1421149, 1421827, 1422537, 1422642, 1422984, 1423028, 1424953, 1425030, 1426217, 1427016, 1427032, 1427656, 1427785, 1429106, 1431382, 1432552, 1433039, 1433781, 1434588, 1434808, 1435261, 1435801, 1436750, 1437451, 1437565, 1437827, 1441437, 1442230, 1443359, 1444005, 1444203, 1444467, 1444981, 1447140, 1448420, 1449916, 1450366, 1451558, 1452151, 1452819, 1453358, 1454065, 1454154, 1454797, 1454821, 1454864, 1455242, 1457572, 1458394, 1459309, 1459431, 1461206, 1461214, 1461257, 1461770, 1462721, 1463359, 1465092, 1465212, 1465872, 1465940, 1469063, 1469211, 1471180, 1471341, 1471560, 1471841, 1472578, 1472873, 1473700, 1473788, 1474221, 1474541, 1476830, 1477475, 1478026, 1478835, 1479136, 1480104, 1481102, 1484358, 1484454, 1484588, 1485557, 1486224, 1486360, 1486813, 1487444, 1488089, 1488658, 1489284, 1489818, 1494128, 1494775, 1495609, 1495983, 1496382, 1496583, 1496829, 1499613, 1499618, 1502502, 1503332, 1503575, 1506786, 1507794, 1508808, 1510287, 1510991, 1512438, 1512542, 1512927, 1513276, 1513670, 1514063, 1515526, 1516166, 1516763, 1517631, 1517847, 1518685, 1520907, 1521464, 1521490, 1522067, 1522182, 1522870, 1523680, 1525023, 1525687, 1527285, 1527518, 1527780, 1527901, 1528211, 1529483, 1530080, 1530727, 1530881, 1531793, 1532940, 1533123, 1533649, 1533776, 1535069, 1537690, 1537968, 1539301, 1541015, 1541350, 1542336, 1542501, 1546812, 1547687, 1548178, 1550936, 1550980, 1551243, 1552760, 1554047, 1555852, 1555999, 1560530, 1561665, 1561674, 1561925, 1562364, 1562663, 1562737, 1563552, 1565052, 1566883, 1568198, 1569927, 1569936, 1569969, 1570093, 1570642, 1573415, 1573881, 1574908, 1574922, 1575008, 1576784, 1577398, 1577824, 1580454, 1580506, 1580671, 1580796, 1581170, 1581478, 1581505, 1581582, 1581706, 1581903, 1583195, 1583355, 1583753, 1584219, 1586587, 1588631, 1589361, 1591189, 1592610, 1593527, 1593728, 1593978, 1594423, 1595151, 1595283, 1595596, 1596043, 1597094, 1597956, 1598230, 1600910, 1601194, 1601271, 1601913, 1602365, 1602790, 1605297, 1605685, 1608396, 1609371, 1609378, 1609416, 1609842, 1610140, 1610839, 1610989, 1611790, 1613717, 1616166, 1616216, 1618896, 1619676, 1619691, 1620444, 1620639, 1621081, 1621130, 1621508, 1622228, 1622847, 1623078, 1623697, 1623963, 1624795, 1625098, 1626804, 1627147, 1627731, 1628089, 1628562, 1630530, 1630988, 1631115, 1631281, 1631482, 1631710, 1632217, 1632287, 1632510, 1633203, 1633590, 1633627, 1634178, 1635184, 1637545, 1639471, 1639724, 1640250, 1641787, 1642612, 1645175, 1645684, 1649216, 1650907, 1651725, 1651830, 1651831, 1652269, 1654313, 1654740, 1655769, 1655996, 1656352, 1656910, 1657012, 1657592, 1657795, 1657934, 1658930, 1658962, 1659932, 1660309, 1660541, 1661598, 1661713, 1662301, 1663133, 1663213, 1666108, 1668634, 1669144, 1669602, 1670623, 1671024, 1671038, 1675198, 1675449, 1676842, 1679011, 1679567, 1680826, 1683127, 1683892, 1684892, 1685041, 1685143, 1686470, 1686588, 1688802, 1689279, 1689566, 1692705, 1697044, 1699748, 1702586, 1703458, 1704658, 1706252, 1706289, 1707401, 1707490, 1708885, 1710198]

paths = [jieba_train_path, jieba_valid_path, jieba_test_path,
         thulac_train_path, thulac_valid_path, thulac_test_path]
for each in paths:
    if not os.path.exists(each):
        os.makedirs(each)
        
def init():
    f = open('../cail_0518/law.txt', 'r', encoding = 'utf8')
    law = {}
    lawname = {}
    line = f.readline()
    while line:
        lawname[len(law)] = line.strip()
        law[line.strip()] = len(law)
        line = f.readline()
    f.close()


    f = open('../cail_0518/accu.txt', 'r', encoding = 'utf8')
    accu = {}
    accuname = {}
    line = f.readline()
    while line:
        accuname[len(accu)] = line.strip()
        accu[line.strip()] = len(accu)
        line = f.readline()
    f.close()


    return law, accu, lawname, accuname


# law, accu, lawname, accuname = init()


def getClassNum(kind):
    global law
    global accu

    if kind == 'law':
        return len(law)
    if kind == 'accu':
        return len(accu)


def getName(index, kind):
    global lawname
    global accuname
    if kind == 'law':
        return lawname[index]
        
    if kind == 'accu':
        return accuname[index]
    

def gettime(time,type=False):
    #将刑期用分类模型来做
    v = int(time['imprisonment'])
#     print(v)

    if time['death_penalty']:
        return -2
    if time['life_imprisonment']:
        return -1
    if type:
        if(v>1):
            return math.log(v)
        else:
            return v
    return v


def get_labels():
    global law
    global accu
    # 做单标签
    """获取训练集所有样本的标签。注意之前在处理数据时丢弃了部分没有 title 的样本。"""
    f = open(sourcefile,mode='r',encoding = 'utf-8')
    line = f.readline()
    index = 0
    law_labellist = []
    accu_labellist = []
    time_labellist = []
    time_labellistlog = []
    while line:
        if(index not in drop_indexlist):
            data = line.replace('\n','')
            data = json.loads(data)
            lawsjson = data['meta']['relevant_articles']
            accusjson = data['meta']['accusation']
            timesjson = data['meta']['term_of_imprisonment']
            lawslabel = []
            accuslabel = []
            for lawitem in lawsjson:
                lawslabel.append(law[str(lawitem)])
            law_labellist.append(lawslabel)
            for accitem in accusjson:
                accuslabel.append(accu[accitem])
            accu_labellist.append(accuslabel)
            time_labellist.append(gettime(timesjson))
            time_labellistlog.append(gettime(timesjson, type=True))
        index +=1
        line = f.readline()
    f.close()
    print(len(drop_indexlist))
    print(len(law_labellist))
    print(len(accu_labellist))
    print(len(time_labellist))
    print(len(time_labellistlog))
    #train:valid:test=6:1:2
    valid_num = 190000
    test_num = 380000
    totalnum = len(law_labellist)
    new_index = np.random.permutation(totalnum)
    time_labellist = np.asarray(time_labellist)[new_index]
    time_labellistlog = np.asarray(time_labellistlog)[new_index]
    law_labellist = np.asarray(law_labellist)[new_index]
    accu_labellist = np.asarray(accu_labellist)[new_index]
    
    valid_law_labellist = law_labellist[:valid_num]
    test_law_labellist = law_labellist[valid_num:test_num+valid_num]
    train_law_labellist = law_labellist[test_num+valid_num:]
    valid_law_np = np.asarray(valid_law_labellist)
    test_law_np = np.asarray(test_law_labellist)
    train_law_np = np.asarray(train_law_labellist)
    np.save(jieba_valid_path+'valid_law_label.npy', valid_law_np)
    np.save(jieba_test_path+'test_law_label.npy', test_law_np)
    np.save(jieba_train_path+'train_law_label.npy', train_law_np)
    
    valid_accu_labellist = accu_labellist[:valid_num]
    test_accu_labellist = accu_labellist[valid_num:test_num+valid_num]
    train_accu_labellist = accu_labellist[test_num+valid_num:]
    valid_law_np = np.asarray(valid_accu_labellist)
    test_law_np = np.asarray(test_accu_labellist)
    train_law_np = np.asarray(train_accu_labellist)
    np.save(jieba_valid_path+'valid_accu_label.npy', valid_law_np)
    np.save(jieba_test_path+'test_accu_label.npy', test_law_np)
    np.save(jieba_train_path+'train_accu_label.npy', train_law_np)
    
    valid_time_labellist = time_labellist[:valid_num]
    test_time_labellist = time_labellist[valid_num:test_num+valid_num]
    train_time_labellist = time_labellist[test_num+valid_num:]
    valid_law_np = np.asarray(valid_time_labellist)
    test_law_np = np.asarray(test_time_labellist)
    train_law_np = np.asarray(train_time_labellist)
    np.save(jieba_valid_path+'valid_time_label.npy', valid_law_np)
    np.save(jieba_test_path+'test_time_label.npy', test_law_np)
    np.save(jieba_train_path+'train_time_label.npy', train_law_np)
    
    valid_time_labellistlog = time_labellistlog[:valid_num]
    test_time_labellistlog = time_labellistlog[valid_num:test_num+valid_num]
    train_time_labellistlog = time_labellistlog[test_num+valid_num:]
    valid_law_np = np.asarray(valid_time_labellistlog)
    test_law_np = np.asarray(test_time_labellistlog)
    train_law_np = np.asarray(train_time_labellistlog)
    np.save(jieba_valid_path+'valid_time_labellog.npy', valid_law_np)
    np.save(jieba_test_path+'test_time_labellog.npy', test_law_np)
    np.save(jieba_train_path+'train_time_labellog.npy', train_law_np)
    
    return new_index

def pad_X400(words, max_len=400):
    """把 jiebatext 整理成固定长度。
    """
    words = list(words)
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[-max_len:]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])


def pad_X350(words, max_len=350):
    """把 thulactext 整理成固定长度。
    """
    words = list(words)
    words_len = len(words)
    words = np.asarray(words)
    if words_len == max_len:
        return words
    if words_len > max_len:
        return words[-max_len:]
    return np.hstack([words, np.zeros(max_len-words_len, dtype=int)])

with open('../data/sr_word2id.pkl', 'rb') as inp:
        sr_id2word = pickle.load(inp)
        sr_word2id = pickle.load(inp)
dict_word2id = dict()
for i in range(len(sr_word2id)):
    dict_word2id[sr_word2id.index[i]] = sr_word2id.values[i]

def get_id(word):
    
    """获取 word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_word2id:
        return 1
    else:
        return dict_word2id[word]
def get_id4words(words):
    """把 words 转为 对应的 id"""
    ids = map(get_id, words)  # 获取id
    return ids

def get_datas_jieba():
    """转换jieba数据集成数组，shape(,400)"""
    print("转换jieba数据集成数组，shape(,400)")
    f = open(filenamesplited,mode='r',encoding = 'utf-8')
    line = f.readline()
    index = 0
    dataarray = np.zeros((1709387,400),dtype = int)
    i = 0
    while line:
        if(index not in drop_indexlist):
            data = line.replace('\n','').split(' ')
            a = get_id4words(data)
            dataarray[i] = np.asarray(pad_X400(a))
            i += 1
        index +=1
        line = f.readline()
    f.close()
    print(index)
    print(i)
    #train:valid:test=6:1:2
    valid_num = 190000
    test_num = 380000
    new_index = np.load('../data/index.npy')
    datanewarray = dataarray[new_index]
    del dataarray,new_index
    print(datanewarray.shape)
    
#     p =  Pool()
#     datalist = np.asarray(p.map(get_id4words, datalist))
#     datalist = np.asarray(p.map(pad_X400, datalist))
    
    valid_data_list = datanewarray[:valid_num]
    np.save(jieba_valid_path+'valid_data.npy', valid_data_list)
    print('--save valid successfully--')
    del valid_data_list
    test_data_list = datanewarray[valid_num:test_num+valid_num]
    np.save(jieba_test_path+'test_data.npy', test_data_list)
    print('--save test successfully--')
    del test_data_list
    train_data_list = datanewarray[test_num+valid_num:]
    np.save(jieba_train_path+'train_data.npy', train_data_list)
    print('--save train successfully--')
    del train_data_list,datanewarray 
#     p.close()
#     p.join()


with open('../data/sr_thulac2id.pkl', 'rb') as inp:
        sr_id2thulac = pickle.load(inp)
        sr_thulac2id = pickle.load(inp)
dict_thulac2id = dict()
for i in range(len(sr_thulac2id)):
    dict_thulac2id[sr_thulac2id.index[i]] = sr_thulac2id.values[i]

def get_id_thulac(word):
    
    """获取 thulac word 所对应的 id.
    如果该词不在词典中，用 <UNK>（对应的 ID 为 1 ）进行替换。
    """
    if word not in dict_thulac2id:
        return 1
    else:
        return dict_thulac2id[word]

def get_id4words_thulac(words):
    """把 words 转为 对应的 id"""
    ids = map(get_id_thulac, words)  # 获取id
    return ids

def get_datas_thulac():
    """转换thulac数据集成数组，shape(,350)"""
    print("转换thulac数据集成数组，shape(,350)")
    f = open(filenamecut,mode='r',encoding = 'utf-8')
    line = f.readline()
    index = 0
    dataarray = np.zeros((1709387,350),dtype = int)
    i = 0
    while line:
        if(index not in drop_indexlist):
            data = line.replace('\n','').split(' ')
            a = get_id4words_thulac(data)
            dataarray[i] = np.asarray(pad_X350(a))
            i += 1
        index +=1
        line = f.readline()
    f.close()
    print(index)
    print(i)
    #train:valid:test=6:1:2
    valid_num = 190000
    test_num = 380000
    new_index = np.load('../data/index.npy')
    datanewarray = dataarray[new_index]
    del dataarray,new_index
    print(datanewarray.shape)
    
    valid_data_list = datanewarray[:valid_num]
    np.save(thulac_valid_path+'valid_data_thulac.npy', valid_data_list)
    print('--save valid thulac successfully--')
    del valid_data_list
    test_data_list = datanewarray[valid_num:test_num+valid_num]
    np.save(thulac_test_path+'test_data_thulac.npy', test_data_list)
    print('--save test thulac successfully--')
    del test_data_list
    train_data_list = datanewarray[test_num+valid_num:]
    np.save(thulac_train_path+'train_data_thulac.npy', train_data_list)
    print('--save train thulac successfully--')
    del train_data_list,datanewarray
    
if __name__=='__main__':
#     a = np.zeros((1800000,400))
#     print(a.shape)
#     get_datas_jieba()
    get_datas_thulac()
    
