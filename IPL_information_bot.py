#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(update, context):
    """Echo the user message."""
    question = update.message.text
    paragraph = '''The Indian Premier League (IPL) is a professional Twenty20 cricket league, contested by eight teams based out of eight different Indian cities.[3] The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. It is usually held between March and May of every year and has an exclusive window in the ICC Future Tours Programme.[4]

The IPL is the most-attended cricket league in the world and in 2014 was ranked sixth by average attendance among all sports leagues.[5] In 2010, the IPL became the first sporting event in the world to be broadcast live on YouTube.[6][7] The brand value of the IPL in 2019 was ₹475 billion (US$6.7 billion), according to Duff & Phelps.[8] According to BCCI, the 2015 IPL season contributed ₹11.5 billion (US$160 million) to the GDP of the Indian economy.[9]

There have been thirteen seasons of the IPL tournament. The current IPL title holders are the Mumbai Indians, who won the 2020 season.[10] The venue for the 2020 season was moved due to the COVID-19 pandemic and games were played in the United Arab Emirates.[11][12] Background
The Indian Cricket League (ICL) was founded in 2007, with funding provided by Zee Entertainment Enterprises.[13] The ICL was not recognised by the Board of Control for Cricket in India (BCCI) or the International Cricket Council (ICC) and the BCCI were not pleased with its committee members joining the ICL executive board.[14] To prevent players from joining the ICL, the BCCI increased the prize money in their own domestic tournaments and also imposed lifetime bans on players joining the ICL, which was considered a rebel league by the board.[15][16]

Foundation
The IPL has been designed to entice an entire new generation of sports fans into the grounds throughout the country. The dynamic Twenty20 format has been designed to attract a young fan base, which also includes women and children.
— Lalit Modi during the launch of the IPL.[17]
On 13 September 2007,[17] on the back of India's victory at the 2007 T20 World Cup,[18] BCCI announced a franchise-based Twenty20 cricket competition called Indian Premier League. The first season was slated to start in April 2008, in a "high-profile ceremony" in New Delhi. BCCI vice-president Lalit Modi, who spearheaded the IPL effort, spelled out the details of the tournament including its format, the prize money, franchise revenue system and squad composition rules. It was also revealed that the IPL would be run by a seven-man governing council composed of former India players and BCCI officials and that the top two teams of the IPL would qualify for that year's Champions League Twenty20. Modi also clarified that they had been working on the idea for two years and that the IPL was not started as a "knee-jerk reaction" to the ICL.[17] The league's format was similar to that of the Premier League of England and the NBA in the United States.[16]

In order to decide the owners for the new league, an auction was held on 24 January 2008 with the total base prices of the franchises costing around $400 million.[16] At the end of the auction, the winning bidders were announced, as well as the cities the teams would be based in: Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata, Mohali, and Mumbai.[16] In the end, the franchises were all sold for a total of $723.59 million.[19] The Indian Cricket League soon folded in 2008.[20]

Expansions and terminations
On 21 March 2010, two new franchises – Pune Warriors India and Kochi Tuskers Kerala – joined the league before the fourth season in 2011.[21] Sahara Adventure Sports Group bought the Pune franchise for $370 million while Rendezvous Sports World bought the Kochi franchise for $333.3 million.[21] However, one year later, on 11 November 2011, it was announced that the Kochi Tuskers Kerala side would be terminated following the side breaching the BCCI's terms of conditions.[22]

Then, on 14 September 2012, following the team not being able to find new owners, the BCCI announced that the 2009 champions, the Deccan Chargers, would be terminated.[23] The next month, on 25 October, an auction was held to see who would be the owner of the replacement franchise, with Sun TV Network winning the bid for the Hyderabad franchise.[24] The team would be named Sunrisers Hyderabad.[25]

Pune Warriors India withdrew from the IPL on 21 May 2013 over financial differences with the BCCI.[26] The franchise was officially terminated by the BCCI, on 26 October 2013, on account of the franchise failing to provide the necessary bank guarantee.[27]

On 14 June 2015, it was announced that two-time champions, Chennai Super Kings, and the inaugural season champions, Rajasthan Royals, would be suspended for two seasons following their role in a match-fixing and betting scandal.[28] Then, on 8 December 2015, following an auction, it was revealed that Pune and Rajkot would replace Chennai and Rajasthan for two seasons.[29] The two teams were the Rising Pune Supergiant and the Gujarat Lions.[30]

Organisation
Tournament format
Currently, with eight teams, each team plays each other twice in a home-and-away round-robin format in the league phase. At the conclusion of the league stage, the top four teams will qualify for the playoffs. The top two teams from the league phase will play against each other in the first Qualifying match, with the winner going straight to the IPL final and the loser getting another chance to qualify for the IPL final by playing the second Qualifying match. Meanwhile, the third and fourth place teams from league phase play against each other in an eliminator match and the winner from that match will play the loser from the first Qualifying match. The winner of the second Qualifying match will move onto the final to play the winner of the first Qualifying match in the IPL Final match, where the winner will be crowned the Indian Premier League champions.

Player acquisition, squad composition and salaries
A team can acquire players through any of the three ways: the annual player auction, trading players with other teams during the trading windows, and signing replacements for unavailable players. Players sign up for the auction and also set their base price, and are bought by the franchise that bids the highest for them. Unsold players at the auction are eligible to be signed up as replacement signings. In the trading windows, a player can only be traded with his consent, with the franchise paying the difference if any between the old and new contracts. If the new contract is worth more than the older one, the difference is shared between the player and the franchise selling the player. There are generally three trading windows—two before the auction and one after the auction but before the start of the tournament. Players cannot be traded outside the trading windows or during the tournament, whereas replacements can be signed before or during the tournament.

Some of the team composition rules (as of 2020 season) are as follows:

The squad strength must be between 18 and 25 players, with a maximum of 8 overseas players.
Salary cap of the entire squad must not exceed ₹850 million (US$12 million).[31]
Under-19 players cannot be picked unless they have previously played first-class or List A cricket.
A team can play a maximum of 4 overseas players in their playing eleven.[32]
The term of a player contract is one year, with the franchise having the option to extend the contract by one or two years. Since the 2014 season, the player contracts are denominated in the Indian rupee, before which the contracts were in U.S. dollars. Overseas players can be remunerated in the currency of the player's choice at the exchange rate on either the contract due date or the actual date of payment.[33] Prior to the 2014 season, Indian domestic players were not included in the player auction pool and could be signed up by the franchises at a discrete amount while a fixed sum of ₹1 million (US$14,000) to ₹3 million (US$42,000) would get deducted per signing from the franchise's salary purse. This received significant opposition from franchise owners who complained that richer franchises were "luring players with under-the-table deals" following which the IPL decided to include domestic players in the player auction.[34]

According to a 2015 survey by Sporting Intelligence and ESPN The Magazine, the average IPL salary when pro-rated is US$4.33 million per year, the second highest among all sports leagues in the world. Since the players in the IPL are only contracted for the duration of the tournament (less than two months), the weekly IPL salaries are extrapolated pro rata to obtain an average annual salary, unlike other sports leagues in which players are contracted by a single team for the entire year.[35]

Match rules
IPL games utilise television timeouts and hence there is no time limit in which teams must complete their innings. However, a penalty may be imposed if the umpires find teams misusing this privilege. Each team is given a two-and-a-half-minute "strategic timeout" during each innings; one must be taken by the bowling team between the ends of the 6th and 9th overs, and one by the batting team between the ends of the 13th and 16th overs.[36]

Since the 2018 season, the Umpire Decision Review System is being used in all IPL matches, allowing each team one chance to review an on-field umpire's decision per innings.[37]

Prize money
The 2019 season of the IPL offered a total prize money of ₹500 million (US$7.0 million), with the winning team netting ₹200 million (US$2.8 million). The first and second runners up received ₹125 million (US$1.8 million) and ₹87.5 million (US$1.2 million), respectively, with the fourth placed team also winning ₹87.5 million (US$1.2 million).[38] The other teams are not awarded any prize money. The IPL rules mandate that half of the prize money must be distributed among the players.[39] Tournament seasons and results
Main articles: List of Indian Premier League seasons and results and List of Indian Premier League records and statistics
Out of the thirteen teams that have played in the Indian Premier League since its inception, one team has won the competition five times, one team has won the competition thrice, one team has won the competition twice and three other teams have won it once. Mumbai Indians are the most successful team in league's history in terms of the number of titles won. The Chennai Super Kings have won 3 titles,[41] the Kolkata Knight Riders have won two titles,[42] and the other three teams who have won the tournament are the Deccan Chargers, Rajasthan Royals and Sunrisers Hyderabad.[43][44][45] The current champions are the Mumbai Indians who defeated the Delhi Capitals by five wickets in the final of the 2020 season securing their fifth title and winning back-to-back championships.[46] Awards
Main article: List of Indian Premier League awards
Orange Cap
The Orange Cap is awarded to the top run-scorer in the IPL during a season. It is an ongoing competition with the leader wearing the cap throughout the tournament until the final game, with the eventual winner keeping the cap for the season.[80] Latest winner - KL Rahul (2020)

Purple Cap
The Purple Cap is awarded to the top wicket-taker in the IPL during a season. It is an ongoing competition with the leader wearing the cap throughout the tournament until the final game, with the eventual winner keeping the cap for the season.[81] Latest winner - Kagiso Rabada (2020)

Most Valuable Player
The award was called the "man of the tournament" till the 2012 season. The IPL introduced the Most Valuable Player rating system in 2013, the leader of which would be named the "Most Valuable Player" at the end of the season. Latest winner - Jofra Archer (2020)

Fairplay Award
The Fair Play Award is given after each season to the team with the best record of fair play. The winner is decided on the basis of the points the umpires give to the teams. After each match, the two on-field umpires, and the third umpire, scores the performance of both the teams. Latest winner - Mumbai Indians (2020)

Emerging player award
The award was presented for the "best under-19 player" in 2008 and "best under-23 player" in 2009 and 2010, being called "Under-23 Success of the Tournament". In 2011 and 2012, the award was known as "Rising Star of the Year", while, in 2013, it was called "Best Young Player of the Season". Since 2014, the award has been called the Emerging Player of the Year. Latest winner- Devdutt Padikkal (2020)

Most sixes award
The Maximum Sixes Award, currently known as Unacademy Let's Crack It Sixes Award for sponsorship reasons, is presented to the batsman who hits the most sixes in a season of the IPL. Latest winner- Ishan Kishan (2020)

Player of the match (final)
Latest winner - Trent Boult (2020)

Financials
Title sponsorship
From 2008 to 2012, the title sponsor was DLF, India's largest real estate developer, who had secured the rights with a bid of ₹200 crore (US$28 million) for five seasons.[82] After the conclusion of the 2012 season, PepsiCo bought the title sponsorship rights for ₹397 crore (US$56 million) for the subsequent five seasons.[83] However, the company terminated the deal in October 2015, two years before the expiry of the contract, reportedly due to the two-season suspension of Chennai and Rajasthan franchises from the league.[84] The BCCI then transferred the title sponsorship rights for the remaining two seasons of the contract to Chinese smartphone manufacturer Vivo for ₹200 crore (US$28 million).[85] In June 2017, Vivo retained the rights for the next five seasons (2018–2022) with a winning bid of ₹2,199 crore (US$310 million), in a deal more expensive than Barclays' Premier League title sponsorship contract between 2013 and 2016.[86][87] On 4 August 2020, Vivo got out of the title sponsorship rights due to the ongoing military stand-off between India and China at the Line of Actual Control in July 2020.[88] It was also reported that the withdrawal was a result of Vivo's market losses due to the ongoing COVID-19 situation and that it intended to return as the title sponsors for the following 3 years.[89] Dream11 bagged the title sponsorship for the 2020 IPL for an amount of ₹220 crore.[90] The tournament has grown rapidly in value over the years 2016–18, as seen in a series of jumps in value from one season to the next. The IPL as a whole was valued by financial experts at US$4.16 billion in 2016, but that number grew to $5.3 billion in 2017, and $6.13 billion in 2018. A report from Duff & Phelps said that one of the contributing factors in the rapid growth of the value of the Indian Premier League was signing a new television deal with Star India Private Limited, which engaged more viewers due to the fact that the IPL was transmitted to regional channels in 8 different languages, rather than the previous deal, which saw the transmissions limited to sports networks with English language commentary.[91][92] The report also stated that the game continued to recover from recent controversy, stating "This IPL season has grabbed the eyeballs for all the right reasons with a relatively controversy free tournament, coupled with some scintillating on-field performances which have brought the spotlight back on the game."[93]

According to another independent report conducted by Brand Finance, a London-based company, after the conclusion of the 2017 Indian Premier League, the IPL has seen its business value grow by 37% to an all-time high of $5.3 billion — crossing the five billion mark for the first time in a season. According to the director of the company: "Now in its 11th season, the Indian Premier League is here to stay. The league has delivered financially for the players, franchisees, sponsors and India as a whole, prompting a strong desire among a range of stakeholders to appropriately value it. To ensure continued development, management and team owners will have to explore innovative ways of engaging fans, clubs, and sponsors."[94] Broadcasting
The IPL's broadcast rights were originally held by a partnership between Sony Pictures Networks and World Sport Group under a ten-year contract valued at US$1.03 billion. Sony would be responsible for domestic television, while WSG would handle international distribution.[95][96] The initial plan was for 20% of these proceeds to go to the IPL, 8% as prize money and 72% would be distributed to the franchisees from 2008 until 2012, after which the IPL would go public and list its shares.[97] However, in March 2010, the IPL decided not to go public and list its shares.[98] As of the 2016 season, Sony MAX, Sony SIX, and Sony ESPN served as the domestic broadcasters of the IPL; MAX and SIX aired broadcasts in Hindi, while SIX also aired broadcasts in the Bengali, Tamil, Kannada and Telugu languages. Sony ESPN broadcast English-language feeds.[99] Sony also produced an entertainment-oriented companion talk show, Extraaa Innings T20, which featured analysis and celebrity guests.[100]

The IPL became a major television property within India; Sony MAX typically became the most-watched television channel in the country during the tournament,[101] and by 2016, annual advertising revenue surpassed ₹12 billion (US$170 million). Viewership numbers were expected to increase further during the 2016 season due to the industry adoption of the new BARC ratings system, which also calculates rural viewership rather than only urban markets.[102][99] In the 2016 season, Sony's broadcasts achieved just over 1 billion impressions (television viewership in thousands), jumping to 1.25 billion the following year.[101]

On 4 September 2017, it was announced that the then-current digital rights-holder, Star India, had acquired the global media rights to the IPL under a five-year contract beginning in 2018. Valued at ₹163.475 billion (US$2.55 billion, £1.97 billion), it is a 158% increase over the previous deal, and the most expensive broadcast rights deal in the history of cricket. The IPL sold the rights in packages for domestic television, domestic digital, and international rights; although Sony held the highest bid for domestic television, and Facebook had made a US$600 million bid for domestic digital rights (which U.S. media interpreted as a sign that the social network was interested in pursuing professional sports rights),[103][104] Star was the only bidder out of the shortlist of 14 to make bids in all three categories.[105][106][107]

Star CEO Uday Shankar stated that the IPL was a "very powerful property", and that Star would "remain very committed to make sure that the growth of sports in this country continues to be driven by the power of cricket". He went on to say that "whoever puts in that money, they put in that money because they believe in the fans of the sport. The universe of cricket fans, it tells you, continues to very healthy, continues to grow. What was paid in 2008, that was 2008. India and cricket and IPL—all three have changed dramatically in the last 10 years. It is a reflection of that."[106][105][107] The deal led to concerns that Star India now held a monopoly on major cricket rights in the country, as it is also the rights-holder of ICC competitions and the Indian national team.[108]

For its inaugural season, Star aimed to put a larger focus on widening the IPL's appeal with a "core" cricket audience. The network aimed to broadcast at least two hours of IPL-related programming daily from January until the start of the season, having organised televised announcements of player retention selections and new team captains. Viewership of the player auction, which featured pre- and post-auction reactions and analysis, increased six-fold to 46.5 million. In March, Star Sports broadcast Game Plan: In Your City specials from the home city of each of the IPL's franchises. Star Sports stated that its in-season coverage and studio programming would focus more on the game itself and behind-the-scenes coverage of the IPL's teams, rather than trying to incorporate irrelevant entertainment elements. The network introduced a new studio program known as The Dugout, which broadcasts coverage of matches with analysis from a panel of experts.[109]

Star broadcasts IPL matches live online in India via its over-the-top video streaming platform Disney+ Hotstar[110] to subscribers of Disney+ Hotstar VIP or Disney+ Hotstar Premium.[111] Matches are also available on Jio TV and Airtel TV apps on smartphones.[112] Throughout the 2019 season, international streaming viewership on Disney+ Hotstar saw records, exceeding 10 million concurrent viewers multiple times. The 2019 final broke these records, peaking at 18.6 million concurrent streaming viewers.[113]

Due to the tournament's popularity, Disney (the owner of Star India) decided to launch their streaming service Disney+ (via Hotstar) in India on 29 March 2020,[114] coincidentally with the beginning of the 2020 season. However, with the postponement of the season due to the COVID-19 pandemic,[115] the service was nonetheless launched on 3 April 2020 with a short delay.[116][117]

International broadcasters '''
    question = '[CLS] ' + question + '[SEP]'
    paragraph = paragraph[:700] + '[SEP]'
    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(paragraph)
    tokens = question_tokens + paragraph_tokens 
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(question_tokens)
    segment_ids += [1] * len(paragraph_tokens)
    input_ids = torch.tensor([input_ids])
    segment_ids = torch.tensor([segment_ids])
    start_scores, end_scores = model(input_ids, token_type_ids = segment_ids)
    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)


    answer = ' '.join(tokens[start_index:end_index+1])
    update.message.reply_text(answer.replace('##',''))


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("1854852527:AAETjdpSu6FZrmx1JVTPIU7nqnMogb-P58s", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
