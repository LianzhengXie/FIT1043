setwd("D:/2022S1/1043/a3/Twitter_Data_1")
Sys.setlocale("LC_TIME", "C")
twitter <- read.csv("a.csv", header = F)
twitter$V1 <- strptime(twitter$V1, format = "%a %b %d %H:%M:%S %z %Y", tz = 'UTC')
hist(twitter$V1,"days",xlab = "time",col = "yellow",freq = T,ylim = c(0,50))


numtwitter <- read.table("b.txt",fill = TRUE, head = FALSE)
names(numtwitter)<- c("number_twitter","id")
max(numtwitter$number_twitter)
hist(numtwitter$number_twitter,breaks = 243,freq = T, xlim = c(0,10))

