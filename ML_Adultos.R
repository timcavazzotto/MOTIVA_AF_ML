#Motiva-AF Machine Learning paper ADULTS
#Create by Timothy Cavazzotto
#Started at 24/06/2023
#Ended at ...


#Packages
library(readxl)
library(synthpop)
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(tidymodels)

#### database ####
Banco_ML_Adultos <- read_excel("Extr_Adultos.xlsx",
                              sheet = "BARREIRAS_N")

dados<- Banco_ML_Adultos
table(dados$Fator1_n, useNA = "always")
##### simulating barreiras data set ####

barreira<-rep(dados$barreira[1:length(dados$barreira)],
             dados$Fator1_n[1:length(dados$Fator1_n)])

g<-rep(dados$g[1:length(dados$g)],
       dados$Fator1_n[1:length(dados$Fator1_n)])

id<-rep(dados$ID_g[1:length(dados$ID_g)],
        dados$Fator1_n[1:length(dados$Fator1_n)])

pais<-rep(dados$PAIS[1:length(dados$PAIS)],
          dados$Fator1_n[1:length(dados$Fator1_n)])


sexo<-rep(dados$SEXO[1:length(dados$SEXO)],
         dados$Fator1_n[1:length(dados$Fator1_n)])


cond<-rep(dados$GRUPO[1:length(dados$GRUPO)],
          dados$Fator1_n[1:length(dados$Fator1_n)])       
  
id_n<-function(n){
      for(i in 1:n){
      id<-abs(rnorm(dados$Fator1_n[i], dados$IDADE_M[i], dados$IDADE_DP[i]))}
return(id)}



idade<-c(id_n(1),	id_n(2),	id_n(3),	id_n(4),	id_n(5),	id_n(6),	id_n(7),	id_n(8),
         id_n(9),	id_n(10),	id_n(11),	id_n(12),	id_n(13),	id_n(14),	id_n(15),	id_n(16),
         id_n(17),	id_n(18),	id_n(19),	id_n(20),	id_n(21),	id_n(22),	id_n(23),	id_n(24),
         id_n(25),	id_n(26),	id_n(27),	id_n(28),	id_n(29),	id_n(30),	id_n(31),	id_n(32),
         id_n(33),	id_n(34),	id_n(35),	id_n(36),	id_n(37),	id_n(38),	id_n(39),	id_n(40),
         id_n(41),	id_n(42),	id_n(43),	id_n(44),	id_n(45),	id_n(46),	id_n(47),	id_n(48),
         id_n(49),	id_n(50),	id_n(51),	id_n(52),	id_n(53),	id_n(54),	id_n(55),	id_n(56),
         id_n(57),	id_n(58),	id_n(59),	id_n(60),	id_n(61),	id_n(62),	id_n(63),	id_n(64),
         id_n(65),	id_n(66),	id_n(67),	id_n(68),	id_n(69),	id_n(70),	id_n(71),	id_n(72),
         id_n(73),	id_n(74),	id_n(75),	id_n(76),	id_n(77),	id_n(78),	id_n(79),	id_n(80),
         id_n(81),	id_n(82),	id_n(83),	id_n(84),	id_n(85),	id_n(86),	id_n(87),	id_n(88),
         id_n(89),	id_n(90),	id_n(91),	id_n(92),	id_n(93),	id_n(94),	id_n(95),	id_n(96),
         id_n(97),	id_n(98),	id_n(99),	id_n(100),	id_n(101),	id_n(102),	id_n(103),	id_n(104),
         id_n(105),	id_n(106),	id_n(107),	id_n(108),	id_n(109),	id_n(110),	id_n(111),	id_n(112),
         id_n(113),	id_n(114),	id_n(115),	id_n(116),	id_n(117),	id_n(118),	id_n(119),	id_n(120),
         id_n(121),	id_n(122),	id_n(123),	id_n(124),	id_n(125),	id_n(126),	id_n(127),	id_n(128),
         id_n(129),	id_n(130),	id_n(131),	id_n(132),	id_n(133),	id_n(134),	id_n(135),	id_n(136),
         id_n(137),	id_n(138),	id_n(139),	id_n(140),	id_n(141),	id_n(142),	id_n(143),	id_n(144),
         id_n(145),	id_n(146),	id_n(147),	id_n(148),	id_n(149),	id_n(150),	id_n(151),	id_n(152),
         id_n(153),	id_n(154),	id_n(155),	id_n(156),	id_n(157),	id_n(158),	id_n(159),	id_n(160),
         id_n(161),	id_n(162),	id_n(163),	id_n(164),	id_n(165),	id_n(166),	id_n(167),	id_n(168),
         id_n(169),	id_n(170),	id_n(171),	id_n(172),	id_n(173),	id_n(174),	id_n(175),	id_n(176),
         id_n(177),	id_n(178),	id_n(179),	id_n(180),	id_n(181),	id_n(182),	id_n(183),	id_n(184),
         id_n(185),	id_n(186),	id_n(187),	id_n(188),	id_n(189),	id_n(190),	id_n(191),	id_n(192),
         id_n(193),	id_n(194),	id_n(195),	id_n(196),	id_n(197),	id_n(198),	id_n(199),	id_n(200),
         id_n(201),	id_n(202),	id_n(203),	id_n(204),	id_n(205),	id_n(206),	id_n(207),	id_n(208),
         id_n(209),	id_n(210),	id_n(211),	id_n(212),	id_n(213))			


dados_simula<-data.frame(g, pais, sexo, cond, idade, barreira)


#Creating_age group
dados_simula["age_group"] = cut(dados_simula$idade, c(0, 20, 30, 40, 50, 60, 70, 80, Inf), 
                          c("<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", ">80"), 
                          include.lowest=TRUE)         

str(dados_simula)

write.csv2(dados_simula, "dados_simula.csv")

dados_simula<-subset(dados_simula, select = -c(idade, g))

########### Pre_process########################################################
#Transform numeric variables as factor
dados_simula[] <- lapply(dados_simula, function(x) { 
  if(is.character(x)) as.factor(x) else x
}) # all variables as.factor

#Transform numeric variables as factor
dados_simula[] <- lapply(dados_simula, function(x) { 
  if(is.numeric(x)) as.factor(x) else x
}) # all variables as.factor

str(dados_simula)

#define one-hot encoding function
dummy <- dummyVars(" ~ .", data= subset(dados_simula, 
                                        select = -c(barreira)))

#perform one-hot encoding on data frame
dados_simula_final <- data.frame(predict(dummy, newdata=dados_simula))
dados_simula_final <- cbind(dados_simula_final, dados_simula$barreira)
str(dados_simula_final)

dados_simula_final[dados_simula$factor1]
names(dados_simula_final)[c(17)] <- 
  c("Barreiras")
# 
# #Transform numeric variables as factor
# dados_simula_final[] <- lapply(dados_simula_final, function(x) { 
#   if(is.numeric(x)) as.factor(x) else x
# }) # all variables as.factor


write.csv2(dados_simula_final, "dados_simula_dummy.csv")

dados<-dados_simula_final
dados_simula_final<-dados

#creating test and train data
set.seed(3456)
trainIndex <- createDataPartition(dados_simula_final$Barreiras, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)
training_df<-dados_simula_final[trainIndex,]
testing_df<-dados_simula_final[-trainIndex,]



