
# ANALIZA PERFORMANSI KLASIFIKACIONIH ALGORITAMA

library(foreign)
library(mice)
library(mda)# fda regresiju
library(randomForest)
library(caret)#train
library(RWeka)
library(psych)
library(lares)
library(xgboost)
library(tibble)
library(Amelia)
library(rpart)
library(factoextra)
library(FactoMineR)
library(devtools)
library(corrplot)
library(ROCR)
library(magrittr)
library(dplyr)# %>%
library(Matrix)
library(class)
library(tidyr)
library(leaps)
library(glmnet)
library(e1071) # svm
library(rpart.plot)
library(psych)
library(stats)
library(MASS)
library(AICcmodavg)
library(car)

tek_seting <- getOption("warn")

options(warn = -1)

# 1. FORMIRANJE  DATA SETA

# Import podataka

stecaj_1<-read.arff(choose.files())# RWeka, file 5
stecaj_2<-read.arff(choose.files())# file 1
stecaj_3<-read.arff(choose.files())# file 2
stecaj_4<-read.arff(choose.files())# file 4, stecaj posle 2 godine

# Kreiranje data seta. Binarna promenjiva class = (0,1). Za kompanije  u stecaju class= 1

samo_stecaj2<-stecaj_2[stecaj_2$class==1,]
samo_stecaj3<-stecaj_3[stecaj_3$class==1,]
samo_stecaj4<-stecaj_4[stecaj_4$class==1,]

stecaj<-rbind(stecaj_1,samo_stecaj2,samo_stecaj3,samo_stecaj4)


# Random preraspodela uzoraka

set.seed(1)

stecaj<-stecaj[sample(1:nrow(stecaj)),]

# Promena imena kolone zavisne binarne promenjive 'class', u 'Stecaj'

colnames(stecaj)[65]<-'Stecaj'

# Provera balansiranosti podataka

round (prop.table( as.table(table(stecaj$Stecaj))),3)


# 2. OPIS PODATAKA

# Descriptivna statistika data seta.
# Mean, Median, Q1, Q3, Min, Max, prisutvo NA's

summary(stecaj)


# 3. ZAMENA NEPOSTOJECIH PODATAKA NAs 

# Graficka provera postojanja NAs i sablona pojavljivanja,primenom  mice package 

md.pattern(stecaj, plot = F) # Sablon pojavljivanja NAs. mice package. 
#1 observed,0 missing. Prva kolaona pokazuje ucestlost paterna, a zadnja broj NA u paternu


missmap(stecaj)# Amelia package 

# Najveci broj nedostajucih vrednosti Attr37, Attr60, Attr64,...

broj_NAs<-sum(is.na(stecaj))

if (broj_NAs != 0) {
  
  cat("Ukupan broj NAs =",broj_NAs,"\n")
  
}

# Provera da li zavisna promenjiva 'Stecaj' ima NA's

if (sum(is.na(stecaj[,65]))==0) { 
  
  cat("Zavisna promenjiva 'Stecaj' nema NA","\n")
}

# Unos zamenskih vrednosti za NAs primenom RF 

stecaj_bezNA<-rfImpute(Stecaj~.,data=stecaj, iter=2)

sum(is.na(stecaj_bezNA))
missmap(stecaj_bezNA)


# 4. PREGLED DISTRIBUCIJA PREDIKTORA 


opis<-describe(stecaj_bezNA) # psych
head(opis)

# Visoke negativne i pozitivne vrednosti skewness  ukazuju na nesimetricnu distribuciju podataka.
# Visoke vrednosti kurtosis oznacavaju Leptokurtske distribucije za koje je karaktereisticno 
# da se vrhovi distribucije tanji i visoki,a repovi zadebljani.
# Skewness izmedju vrednosti -2 and 2 i kurtosis between -7 and 7, normal distribution (Bryne,2010)

normal_prediktor<-subset(opis,(skew<=2&skew>=-2)&(kurtosis<7&kurtosis>-7),select=c(skew,kurtosis))
normal_prediktor

cat("Prediktori sa normalnom distribucijom su  ",row.names(normal_prediktor)[2],"\n")

hist(stecaj_bezNA$Attr29, main="Histogram varijable X29", xlab="X29")


# 5. PROVERA POSTOJANJA MULTIKOLINEARNOSTI 


corelacija<-cor(stecaj_bezNA[,2:64])# korelacija prediktora
corrplot(corelacija,method='number',type="upper",order="hclust",tl.col='black',tl.pos='n') # plava boja, pozitivna korelacija, crvena negativna. Intezitet boje odgovara intezitetu lin zavisnosti

corr_cross(stecaj_bezNA[,2:64], max_pvalue = 0.05, top = 10)# package lares. Parovi 10 znacajnih korelacija(p-value<0.05) U plavoj boji su pozitivne korelacije. Korelacije vece od 5%

# Sledi da postoji korelacija (pozitivna i negativna),izmedju prediktora

# Graficki prikaz korelacija  (primer za prvih deset prediktora)

pairs.panels(stecaj_bezNA[,2:11], gap=0, bg=c("red","blue")[stecaj_bezNA$Stecaj],pch=21)


# 6. ANALIZA POSTOJANJA EKSTREMNIH VREDNOSTI  

# Vrednosti prediktora koje se nalaze van granice min i max whisker-a, ce se smatrati outliers i bice zamenjenjene
# sa median vrednosti odgovarajuceg prediktora

stecaj_bezekstrema_bezNA<-stecaj_bezNA

for(i in 2:65){
  
  out<-boxplot.stats(stecaj_bezNA[,i],coef = 1.5)$out
  out_ind <- which(stecaj_bezNA[,i] %in% c(out))
  stecaj_bezekstrema_bezNA[c(out_ind),i]<-median(stecaj_bezNA[,i], na.rm = TRUE)
}

summary(stecaj_bezekstrema_bezNA)
 

# Kako je jedino prediktor X55, vrednost Working capital, dat u svojoj  apsolutnoj vrednosti, a ne kao 'ratio'
# pristupicemo  normalizaciji podataka, svodjenjem na isti rang vrednosti ( rang od 0 do 1)

# Min max normalizacija

preproc <- preProcess(stecaj_bezekstrema_bezNA[,-1], method=c("range"))# default range 0-1
stecaj_bezekstrema_bezNA_scaled <- predict(preproc, stecaj_bezekstrema_bezNA[,-1])
stecaj_bezekstrema_bezNA_scaled$Stecaj<- as.factor(stecaj_bezekstrema_bezNA$Stecaj)


# 7.  KREIRANJE TRENING/TESTING PODATAKA (podaci bez NA, ekstremnih vrednosti i standardizovani mean nula i sd 1)


# Train/test podaci u odnosu 70/30

set.seed(2)

split <- createDataPartition (stecaj_bezekstrema_bezNA_scaled$Stecaj, p = .7, list = F)# caret
train <- stecaj_bezekstrema_bezNA_scaled[split,]# Attr1....Attr64, Stecaj
test  <- stecaj_bezekstrema_bezNA_scaled[-split,]


# Proporcionalnost podataka razlicitih klasa

prop.table(table(train$Stecaj))
prop.table(table(test$Stecaj))


# 7. KREIRANJE PRINCIPALNIH KOMPONENTI pc.

train_pc <- stecaj_bezekstrema_bezNA[split,]# Stecaj, Attr1 ....Attr64
test_pc  <- stecaj_bezekstrema_bezNA[-split,]


pc<-prcomp(train_pc[,-1],center=T, scale=T) # kreiranje PC1...PC64, u funkciji Attr1....Attr64

summary(pc)$importance[3,1:40][40]# Kumulativna varijacija prvih 30 PC's (Cumulative proportion)

pc$rotation  # linerna transformacija prediktora u PC. PC= f(Attr1.....Attr64)

pc$x # uzorci izrazeni u vrednostima PC, sada su prediktori PC1...PC64

#  Korelacija izmedju PC je nula, nema linearn zavisnosti

corelacija_pc<-cor(pc$x)
corrplot(corelacija_pc,type="full",order="hclust",tl.col='black',tl.pos='n') # plava boja, pozitivna korelacija, crvena negativna. Intezitet boje odgovara intezitetu lin zavisnosti

#Train/test podaci u funkciji PCs

predict_pctrain<-predict(pc, train_pc[,-1])[,1:40]# ne potrebno to je isto kao pc$x

train_pc<-data.frame(predict_pctrain, Stecaj=train_pc$Stecaj)

predict_pctest<-predict(pc,test_pc[,-1])[,1:40]

test_pc<-data.frame(predict_pctest, Stecaj=test_pc$Stecaj)


# 8. PREDIKTIVNI MODELI


# 8.1 LOGISTICKA REGRESIJA

# Kreiranje modela

model_lr<-glm(Stecaj~ ., data=train, family = binomial)
summary(model_lr)


# Predikcija

predict_lr<-model_lr%>%predict(test,type="response")

# Tacnost modela

predicted_lr<-ifelse(predict_lr>0.3,"1","0")
confusionMatrix(factor(predicted_lr), test$Stecaj, positive = '1')

# Model_lr samo sa znacajnim prediktorima

model_lr1<-glm(Stecaj ~ Attr21 + Attr24 + Attr25 + Attr37 + Attr38 +Attr41 +Attr46 +Attr55 , data=train, family = binomial)
summary(model_lr1)


# Multikolinearnost, ne utice na tacnost klasifikacionog  modela, ali utice na vrednost koeficijenata i tako odnos izmedju outputa i nezavisne promenjive
# Vrednosti  Variance inflation factor (VIF) u rangu 1-5 oznacavaju srednji nivo kolinearnosti

vif_prediktora<-data.frame(vif(model_lr1)) # car paket. VIF prediktora pokazuje koliko dobro se taj prediktor moze 'objasniti' drugim prediktorom
# Moze se pojaviti greska kada su dve ili vise varijabli mnogo (ili perfektno) correlated.

vif_prediktora<-rownames_to_column(vif_prediktora)

max_koli<-vif_prediktora[order(vif_prediktora$vif.model_lr1., decreasing = T),]
max_koli[which(max_koli$vif.model_lr1>5),]


# Predikcija

predict_lr1<-model_lr1%>%predict(test,type="response")


# Tacnost modela

predicted_lr1<-ifelse(predict_lr1>0.3,"1","0")
confusionMatrix(factor(predicted_lr1), test$Stecaj, positive = '1')



# Poredjenje log reg modela prema AIC

# Model koji je vise od 2 AIC (Akaike iformation criteria, pokazuje koliko dobro model opisuje podatke) jedinice nizi od drugog modela , je statisticki znacajno bolji

models<-list(model_lr,model_lr1)
model.names<-c('model_lr','modela_lr1')
aictab(cand.set = models, modnames = model.names)



# Najvazniji, najinformativniji prediktori

coef(model_lr1)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(coef(model_lr1))# uklanjam intercept
df_regkoeficijenti<-rownames_to_column(data.frame(df_regkoeficijenti))# package tibble
df_regkoeficijenti<-df_regkoeficijenti[-1,]
colnames(df_regkoeficijenti)<-c('Prediktori','RegKoeficijenti')
df_regkoeficijenti[order(-abs(df_regkoeficijenti$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# 8.1.1 Log regresija sa PC

model_lrpc<-glm(Stecaj~ ., data=train_pc, family = binomial)#model sa PC
summary(model_lrpc)

# Predikcija

predict_lrpc<-model_lrpc%>%predict(test_pc,type="response")# sa PC

# Tacnost sa PC

predicted_lrpc<-ifelse(predict_lrpc>0.3,"1","0")
confusionMatrix(factor(predicted_lrpc), test_pc$Stecaj, positive = '1')



# 8.1.1 PENALIZED LOGISTIC REGRESSION, regularizacija modela (ne proredimo prema AIC)

#Lasso  regresija

# Kreiranje matrice prediktora 

x_train<- model.matrix(Stecaj~., data=train,)[,-1]# U slucaju postojanja factor promenjivih, kreiraju se dummy promenjive
x_test<-model.matrix(Stecaj~., test,)[,-1]

y_output<-train$Stecaj

# Koriscenjem CV trazimo najoptimalniju vrednost lambda, kojom se odredjuje stepen 'sankcionisanja' kompleksnosti modela tj. umanjenje reg koeficijenata 

lasso_param<-cv.glmnet(x_train,y_output,alpha=1,family="binomial")# glmnet package.  Za alpha 0=1, radi se o Lasso reg

# Kreiranje modela

model_lrlasso<- glmnet(x_train,y_output, alpha = 1, family = "binomial",lambda = lasso_param$lambda.min)

# Predikcija

predict_lrlasso<-model_lrlasso%>%predict(newx=x_test,type="response")

# Tacnost modela
predicted_lrlasso<-ifelse(predict_lrlasso>0.3,"1","0")
confusionMatrix(factor(predicted_lrlasso), test$Stecaj, positive = '1')

# Najvazniji, najinformativniji prediktori

coef(model_lrlasso)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(summary(coef(model_lrlasso)))# uklanjam intercept
df_regkoeficijenti[,1]<- df_regkoeficijenti[,1]-1
df_regkoeficijenti<-df_regkoeficijenti[-1,]

df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# Ridge regresija

ridge_param<-cv.glmnet(x_train,y_output,alpha=0,family="binomial")# glmnet package.  Za alpha =0 , radi se o Ridge reg

# Kreiranje modela

model_lrridge<- glmnet(x_train,y_output, alpha = 0, family = "binomial",lambda = ridge_param$lambda.min)

# Predikcija

predict_lrridge<-model_lrridge%>%predict(newx=x_test,type="response")

# Tacnost modela

predicted_lrridge<-ifelse(predict_lrridge>0.3,"1","0")
confusionMatrix(factor(predicted_lrridge), test$Stecaj, positive = '1')

# Najvazniji, najinformativniji prediktori

coef(model_lrridge)# Intercept i 64 prediktora

df_regkoeficijenti<-data.frame(summary(coef(model_lrridge)))# uklanjam intercept
df_regkoeficijenti[,1]<- df_regkoeficijenti[,1]-1
df_regkoeficijenti<-df_regkoeficijenti[-1,]

df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# Elastic net regresija

elasticnet_param<-cv.glmnet(x_train,y_output,alpha=0.7,family="binomial")# glmnet package. Za alpha izmedju 0 i 1, radi se o Elastic Net regresiji

# Kreiranje modela

model_elasnet<- glmnet(x_train,y_output, alpha = 0.7, family = "binomial",lambda = elasticnet_param$lambda.min)

# Predikcija

predict_elasnet<-model_elasnet%>%predict(newx=x_test,type="response")

# Tacnost modela

predicted_elasnet<-ifelse(predict_elasnet>0.3,"1","0")
confusionMatrix(factor(predicted_elasnet), test$Stecaj, positive = '1')

# Najvazniji, najinformativniji prediktori

coef(model_elasnet)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(summary(coef(model_elasnet)))# uklanjam intercept
df_regkoeficijenti[,1]<- df_regkoeficijenti[,1]-1
df_regkoeficijenti<-df_regkoeficijenti[-1,]

df_regkoeficijenti[order(-abs(df_regkoeficijenti$x)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# 8.1.2 LDA i QDA (koriscenjem prediktora najtacnijeg  lasso regresionog modela)


# LDA

model_lda<-lda(Stecaj~ . -Attr18 -Attr7 -Attr14 -Attr52 -Attr32 -Attr10 -Attr2 -Attr17 -Attr8 -Attr54 , data=train)
model_lda

par(mar=c(1,1,1,1))
plot(model_lda)# x osa je LD1, prvi grafik za klasu 0 , drugi klasu 1

# Predikcija

predict_lda<-model_lda%>%predict(test)
names(predict_lda)# class uzorka, posterior pr da uyorak pripda nekoj od klasa, x su LD
head(predict_lda$posterior,2)


# Tacnost modela

predicted_lda<-ifelse(predict_lda$posterior[,2] >0.3,"1","0")

confusionMatrix(factor(predicted_lda), test$Stecaj, positive = '1')

# Najvazniji, najinformativniji prediktori


coef(model_lda)# Intercept i 64 prediktora

df_regkoeficijenti<-data.frame(coef(model_lda))
impattr_lda<-rownames_to_column(df_regkoeficijenti)# package tibble
colnames(impattr_lda)<-c('Prediktori','RegKoeficijenti')
impattr_lda[order(-abs(impattr_lda$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# QDA

model_qda<-qda(Stecaj~ . -Attr18 -Attr7 -Attr14 -Attr52 -Attr32 -Attr10 -Attr2 -Attr17 -Attr8 -Attr54 , data=train)
model_qda

# Predikcija

predict_qda<-model_qda%>%predict(test)

# Tacnost modela

predicted_qda<-ifelse(predict_qda$posterior[,2] >0.3,"1","0")

confusionMatrix(factor(predicted_qda), test$Stecaj, positive = '1')



# 8.2 STABLA ODLUCIVANJA

# Kreiranje modela

model_tree<-rpart(Stecaj~., data=train, method = "class",control=rpart.control(cp=.0001)) # rpart koristi Ginin index kao meru imfomativnosti prediktora
model_tree # zvezdica oznaka terminalnog noda, simbol ispred zagrade je sta taj nod predvidja ili sta bi bilo predvidjanje kad bi bio terminalni, broj posle uslova je broj uzorka

# Pruning stabla sa Optimalnom vrednosti cp,  odnosno optimalne dubine (kompleksnosnti) stabla.

plotcp(model_tree) 
best_cp <- model_tree$cptable[which.min(model_tree$cptable[,"xerror"]),"CP"]
cat("Optimalna vrednost cp je ",best_cp,"\n")

model_tree1<-prune(model_tree,cp=best_cp)# ako stavim 0.01 stablo ce biti isto kao u modelu model_tree.Biram manje kompleksno stablo
model_tree1

# Model stabla nakon pruninga
rpart.plot(model_tree1)

#  5 najinfoirmativnijih prediktora sa najvecim  mean decrease accuracy 

impattr_tree1<-data.frame(model_tree1$variable.importance)
impattr_tree1<-rownames_to_column(impattr_tree1)# package tibble

colnames(impattr_tree1)<-c('Prediktori','RegKoeficijenti')
impattr_tree1[order(-abs(impattr_tree1$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija primenom DT

predict_tree<-data.frame(predict(model_tree1,test,type="prob"))

# Tacnost modela

predicted_tree <- ifelse(predict_tree[,2] > 0.3, 1,0)
confusionMatrix(factor(predicted_tree),
                test$Stecaj,positive = "1")


# 8.3 RANDOM FOREST


trControl=trainControl("cv",number=5)# repeates, izostavljeno zbog performansi

# Kreiranje modela

model_rf<-train(Stecaj~., data=train,method="rf",trControl= trControl,importance=TRUE, search='random')# Importance, vaznost atributa 
plot(model_rf)

model_rf$bestTune 
cat("Optimalan broj prediktora koji ce se koristiti pri izradi stabala je" ,model_rf$finalModel$mtry )

model_rf$finalModel 

# 5 najinefirmativnijih prediktora sa najvecim  mean decrease accuracy 

imp_prediktora<-as.data.frame(randomForest::importance(model_rf$finalModel))

imp_prediktora[order(imp_prediktora$MeanDecreaseAccuracy, decreasing = T),][1:5,] 

# Graficki prikaz informativnosti prediktora

varImpPlot(model_rf$finalModel,type=1, cex=0.5,n.var=min(5, nrow(model_rf$importance)), main = "Informativnost prediktora")# Mean decrease accuracy, koliko se smanji tacnost modela ako se dati prediktor izostavi

# Predikcija primenom RF modela

predict_rf<- predict(model_rf, test,type = "prob")

# Tacnost modela

predicted_rf <- ifelse(predict_rf$`1`> 0.3, 1,0)
confusionMatrix(factor(predicted_rf),
                test$Stecaj,positive = "1")



# 8.4 BOOSTING MODEL,  xgboost sa xgbTree. Koristi sve corove procesora, paralelno procesiranje


trControl=trainControl("cv",number=5)

# Kreiranje modela, xgboost package. Stohastic gradient boosting 

model_xgboost<-train(Stecaj~., data=train,method="xgbTree",trControl=trControl,base_score=0.3)

# Optimalni parametri modela, nrounds je broj iteracija, eta learning rate, 
#gamma nula sledi nema regularizacije , subsumple je br. uzoraka za svako stablo, ako jedan svi

model_xgboost$bestTune

impattr_xgb<-data.frame(varImp(model_xgboost)[[1]])
impattr_xgb<-rownames_to_column(impattr_xgb)

# Najznacajniji prediktori za tacnost modela
impattr_xgb[(1:5),1]

# Predikcija xgbTree

predict_xgboost<-predict(model_xgboost,test,type = "prob")

# Tacnost modela

predicted_xgboost <- ifelse(predict_xgboost$`1`> 0.3, 1,0)
confusionMatrix(factor(predicted_xgboost),
                test$Stecaj,positive = "1")


# Trazenje optimalne vrednosti granicne vrednosti verovatnoce stecaja, za rad kako bi algorimi bili uporedivi uzete je ista vrednost
actuals<-test$Stecaj

perf1<-performance(prediction(predicted_xgboost, actuals),"tpr", "fpr") # ROCR

plot(perf1,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))

# 8.4.1 xgboost

#define predictor and response variables in training set

train_x = data.matrix(train[, -65])
train_y =as.numeric(train[,65])-1

#define predictor and response variables in testing set
test_x = data.matrix(test[, -65])
test_y = as.numeric(test[,65])-1

#define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

head(getinfo(xgb_train,'label'))
head(getinfo(xgb_test,'label'))


# Kreiranje modela

watchlist = list(train=xgb_train, test=xgb_test)#
train_par<- xgb.train(data = xgb_train, max.depth = 3, nrounds = 100, watchlist = watchlist)#training and testing root mean squared error for each round.

model_xgb<- xgboost(data = xgb_train, max.depth = 3, nrounds = 100, objective = "binary:logistic")# svi prediktori moraju biti numericki vektori

#  5 najinfoirmativnijih prediktora sa najvecim  mean decrease accuracy 

importance_matrica = xgb.importance(colnames(xgb_train), model = model_xgb)
importance_matrica[1:5,][,1]

xgb.plot.importance(importance_matrica[1:5,])

# Predikcija

predict_xgboost<-predict(model_xgb, newdata=xgb_test)

# Tacnost modela

predicted_xgboost <- ifelse(predict_xgboost> 0.3, 1,0)
confusionMatrix(factor(predicted_xgboost),
                test$Stecaj,positive = "1")

# 8.5 NB

# Kreiranje modela

model_nb<-naiveBayes(Stecaj~., data=train)
model_nb # vidimo mean i sd za svaki numericki prediktor razlicitih klasa

train%>%filter(Stecaj=="1")%>%summarise(mean(Attr1),sd(Attr1))

# Predikcija

predict_nbp<-predict(model_nb,test,type='raw')[,2]

# Tacnost modela

predicted_nb <- ifelse(predict_nbp> 0.3, 1,0)

confusionMatrix(factor(predicted_nb),
                test$Stecaj,positive = "1")


# Model sa PC

model_nbpc<-naiveBayes(Stecaj~., data=train_pc)

# Predikcija

#predict_nbpc<-predict(model_nbpc,test_pc), daje klase
predict_nbpcp<-predict(model_nbpc,test_pc,type='raw')[,2]


# Tacnost modela

predicted_nbp <- ifelse(predict_nbpcp> 0.3, 1,0)

confusionMatrix(factor(predicted_nbp),
                test_pc$Stecaj,positive = "1")# cutoff 0.3



# 8.6 KNN model


#kreiranje sintaticka ispravnih imena, za vrednosti karakter vektora

levels(train$Stecaj)<-make.names(levels(train$Stecaj))
levels(test$Stecaj)<-make.names(levels(test$Stecaj))

# KNN 1, koristimo Accurancy za izbor optimalnog broja k

# Primena cv, za trazenje optimalne vrednosti k

trControl<-trainControl(method = "repeatedcv", number = 5,repeats=3)

# Kreiranje modela 

model_knn1<-train(Stecaj~., data = train, method='knn', 
                  trControl=trControl,
                  tuneLength=10 )# teuneLength broj mogucih vrednosti  K 


model_knn1$bestTune# optimalna vrednost suseda k

plot(model_knn1,xlab="Broj suseda K",ylab='Tacnost u funkciji od K', type='b',col='red',lwd=1.5,pch='o')

# Najznacajniji prediktori za tacnost modela

impattr_knn1<-data.frame(varImp(model_knn1)[[1]])
impattr_knn1<-rownames_to_column(impattr_knn1)# package tibble

colnames(impattr_knn1)<-c('Prediktori','RegKoeficijenti')
impattr_knn1[order(-abs(impattr_knn1$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

#Predikcija

predict_knn1<-predict(model_knn1,test, type='prob')[,2] # X1 stecaj


# Tacnost modela

predicted1 = ifelse(predict_knn1 > 0.3, 1,0)

confusionMatrix(factor(predicted1),
                factor(actuals), positive = "1")


# 8.6.1 KNN2, koristimo ROC za izbor k

trControl<-trainControl(method = "repeatedcv", number = 5,repeats=3,classProbs = T,summaryFunction = twoClassSummary)

# Kreiranje modela 

model_knn2<-train(Stecaj~., data = train, method='knn',tuneLength=10, 
                  trControl=trControl,metric="ROC")

plot(model_knn2,xlab="Broj suseda K",ylab='Tacnost u funkciji K', type='b',col='black',lwd=1.5,pch='o')

model_knn2$bestTune

# Vaznost prediktora


impattr_knn2<-data.frame(varImp(model_knn2)[[1]])
impattr_knn2<-rownames_to_column(impattr_knn2)# package tibble

colnames(impattr_knn2)<-c('Prediktori','RegKoeficijenti')
impattr_knn2[order(-abs(impattr_knn2$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_knn2<-predict(model_knn2,test,type='prob')[,2]


# Trazenje optimalne vrednosti granicne vrednosti verovatnoce stecaja


perf2<-performance(prediction(predict_knn2, actuals),"tpr", "fpr")

plot(perf2,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))

# Tacnost modela, za usvojenu granicunu vrednost verovatnoce 0.3 za pozitivan ishod, kako bi modeli bili uporedivi


predicted_knn2 <- ifelse(predict_knn2 > 0.3, 1,0)
confusionMatrix(factor(predicted_knn2),
                factor(actuals),positive = "1")


# 8.7  SUPORT VECTOR MACHINE linear

trControl<-trainControl(method = "cv", number = 5, summaryFunction=twoClassSummary,classProbs=TRUE)#za izbor modela na bazi ROC

# Kreiranje  modela, SVM sa linearnom Kernel funkcijom

model_svmlin <- train(Stecaj ~., data = train, method = "svmLinear", trControl=trControl,metric="ROC")

# Prikaz modela

model_svmlin # c=1, za lin metod.Ova konstanta predstavlja cost, misklasifikacije. Tako se sa vecom
# vrednosti konstante C, verovatnoca pogresne klasifikacije je manja.


impattr_svmlin<-data.frame(varImp(model_svmlin)[[1]])
impattr_svmlin<-rownames_to_column(impattr_svmlin)

# Najznacajniji prediktori za tacnost modela

impattr_svmlin[order(-abs(impattr_svmlin[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 


# Predikcija

predict_svmlin<-predict(model_svmlin,test,type='prob')[,2]

# Optimalne  cutoff vrednosti verovatnoce

perf_svmlin<-performance(prediction(predict_svmlin, actuals),"tpr", "fpr")

plot(perf_svmlin,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))


# Tacnost modela, za usvojenu granicnu vrednost verovatnoce 0.3, za pozitivan ishod

predicted_svmlin <- ifelse( predict_svmlin> 0.3, 1,0)

confusionMatrix(factor(predicted_svmlin),
                factor(actuals),positive = "1")


# 8.7.1  SVM Radial.  Kreiranje modela, SVM sa nelinearnom kernel funkcijom (Radial)

model_svmradial <- train(Stecaj ~., data = train, method = "svmRadial", trControl=trControl, tuneLength = 5,metric="ROC")

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmradial$bestTune

# Prikaz modela

plot(model_svmradial)


impattr_svmradial<-data.frame(varImp(model_svmradial)[[1]])
impattr_svmradial<-rownames_to_column(impattr_svmradial)

# Najznacajniji prediktori za tacnost modela

impattr_svmradial[order(-abs(impattr_svmradial[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmradial<-predict(model_svmradial,test,type='prob')[,2]

# Tacnost modela, za usvojenu granicunu vrednost verovatnoce 0.3 za pozitivan ishod

predicted_svmradial <- ifelse( predict_svmradial> 0.3, 1,0)

confusionMatrix(factor(predicted_svmradial),
                factor(actuals),positive = "1")


# 8.7.2 SVM Poli

# Kreiranje modela, SVM sa nelinearnom kernel funkcijom (polinomalna)

model_svmpoli <- train(Stecaj ~., data = train, method = "svmPoly", metric="ROC",trControl=trControl)

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmpoli$bestTune

# Prikaz modela
plot(model_svmpoli)

impattr_svmpoli<-data.frame(varImp(model_svmpoli)[[1]])
impattr_svmpoli<-rownames_to_column(impattr_svmpoli)

# Najznacajniji prediktori za tacnost modela

impattr_svmpoli[order(-abs(impattr_svmpoli[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmpoli<-predict(model_svmpoli,test,type='prob')[,2]

# Tacnost modela, za usvojenu granicunu vrednost verovatnoce 0.3 za pozitivan ishod


predicted_svmpoli <- ifelse( predict_svmpoli> 0.3, 1,0)
confusionMatrix(factor(predicted_svmpoli),
                factor(actuals),positive = "1")



# 9.  Z SCORE  predikcija 

# Podaci za z score, tacnost predikcije,  nisu scaled

test_noscale<-stecaj_bezekstrema_bezNA[-split,c(1,4,7,8,9,10)]


# X3 je working capitalk/TA, b je  X6 je retain earn/TA, c je X7 je ebita/TA, d je X8 je book equity/TL i e je X9 je sales/TA
# z funkci  ce vratit 1 za kompanije u stecaju

z_funkcija<-function(x){
  
  a=x[1]
  b=x[2]
  c=x[3]
  d=x[4]
  e=x[5]
  z=(1.2*a+1.4*b+3.3*c+0.6*d+1*e)
  
  if (z<1.81) {
    
    return(1)# komanija u stecaju
    
  }else {
    
    return(0) # 0 oznacava komanije za koje po modelu ne proizilazi stecaj ili se ne moze odrediti
  }
  
}


z_score<-apply(test_noscale[,-1],1,z_funkcija)

table(test_noscale$Stecaj,factor(z_score))

confusionMatrix(test_noscale$Stecaj,
                factor(z_score),positive = "1")


# B emerging market

z_funkcija<-function(x){
  
  a=x[1]
  b=x[2]
  c=x[3]
  d=x[4]
  
  z=(3.25 + 6.56*a+3.26*b+6.72 *c+1.05*d)
  
  if (z<1.1) {
    
    return(1)# kompanija u stecaju
    
  }else {
    
    return(0) # 0 oznacava komanije za koje po modelu ne proizilazi stecaj ili se ne moze odrediti
  }
  
}



z_score<-apply(test_noscale[, -c(1,6)],1,z_funkcija)

table(test_noscale$Stecaj,factor(z_score))

confusionMatrix(test_noscale$Stecaj,
                factor(z_score),positive = "1")

#######################################################################



# Vracanje warn messages
options(warn = tek_seting)

# Kraj
