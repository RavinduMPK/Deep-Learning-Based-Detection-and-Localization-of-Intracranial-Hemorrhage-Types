%%
%% This is file `isecure.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% esl.dtx  (with options: `package,isecure,ONECOL,DEEPLIST')
%%
%% isecure.cls Copyright (C) 2008-2018 ISeCure
%%
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%%
\newenvironment{TabularX}[2][1]
  {\def\arraystretch{1.5} \def\tabcolsep{0pt} \tabular[#1]{#2}}
  {\endtabular}

\def\readRCS$#1: #2 #3 #4 #5${%
 \def\RCSfile{#2}%
 \def\RCSversion{#3}%
 \def\RCSdate{#4}%
}
\def\@shortjnl{\relax}
 \def\esp@fileversion{0.4}
 \def\esp@filedate{2010/01/16}
 \def\@journal{ISeCure}
 \def\@company{ISC}
 \def\@issn{2008-2045}
 \def\@journalsite{http://isecure-journal.org}
 \def\@publishmonth{January}
 \def\@publishyear{2011}
 \def\@shortjid{ISeCure}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
%\ProvidesClass{\@shortjid}[\esp@filedate, \esp@fileversion: \@journal]
\usepackage{graphicx}
\usepackage{pifont}
\RequirePackage[hyphens]{url}
\RequirePackage[table]{xcolor}
\RequirePackage{calc}

\def\company#1{\gdef\@company{#1}} \def\@company{ISC}
\def\@copyrightyear{\@publishyear}

\newif\if@witharticletype
\newif\if@seceqn
\newif\if@secthm
\newif\if@nameyear
\newif\if@Elproofing

\@Elproofingfalse
\DeclareOption{12pt}{}
\DeclareOption{11pt}{}
\DeclareOption{10pt}{}
\@twosidetrue  \@mparswitchtrue
\newif\if@narrowdisplay
\@narrowdisplayfalse
\DeclareOption{narrowdisplay}{\@narrowdisplaytrue}
\newif\if@draft\@draftfalse
\DeclareOption{draft}{%
  \@drafttrue
  \overfullrule 5\p@                  % to indicate overfull boxes
}
\DeclareOption{final}{%
  \@draftfalse
  \overfullrule \z@
}
\let\snm\relax \let\cty\relax \let\cny\relax\let\jr\relax
\newif\if@ussrhead \@ussrheadfalse
\DeclareOption{ussrhead}{\@ussrheadtrue}
\@seceqnfalse                             % Default: equation numbering is not
\DeclareOption{seceqn}{\@seceqntrue}      % reset at beginning of each section
\@witharticletypefalse
\DeclareOption{WithArticleType}{\@witharticletypetrue} % Option to have the article type to be written on top of the title
\@secthmfalse
\DeclareOption{Secthm}{\@secthmtrue}
\@nameyearfalse
\DeclareOption{nameyear}{\@nameyeartrue}
\def\@blstr{1}
\DeclareOption{doublespacing}{\def\@blstr{1.7}}
\DeclareOption{reviewcopy}{\def\@blstr{1.7}}
\newif\if@symbold\@symboldfalse
\DeclareOption{symbold}{\@symboldtrue}
\newif\if@amsthm\@amsthmfalse
\DeclareOption{amsthm}{\@amsthmtrue}
 \ExecuteOptions{oneside}
 \ExecuteOptions{onecolumn,final}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Define whether you are dealing with
%         the online copy or the print copy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@Print
\@Printfalse
\DeclareOption{Print}{\@Printtrue}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newif\if@TwoColumn
\@TwoColumntrue
\DeclareOption{OneColumn}{\@TwoColumnfalse}

\ProcessOptions

\definecolor{BLACK}{cmyk}{0,0,0,1}      % CMYK: black

\if@Print
    \definecolor{darkred}{rgb}{0,0,0}     % Web color: Maroon
    \definecolor{darkgreen}{rgb}{0,0,0}   % Web color: Green
    \definecolor{darkblue}{rgb}{0,0,0}    % Web color: Navy
\else
    \definecolor{darkred}{rgb}{0.5,0,0}     % Web color: Maroon
    \definecolor{darkgreen}{rgb}{0,0.5,0}   % Web color: Green
    \definecolor{darkblue}{rgb}{0,0,0.5}    % Web color: Navy
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\if@Print
    \def\ColorDir{CMYK}
\else
    \def\ColorDir{RGB}
\fi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\if@TwoColumn\@narrowdisplayfalse\fi
\if@draft\else\@narrowdisplayfalse\fi
\def\@pagenumprefix{}
\def\author@font{}
\def\partname{Part}
\def\appendixname{Appendix}
\def\contentsname{Contents}
\def\listfigurename{List of Figures}
\def\listtablename{List of Tables}
\def\refname{References}
\def\indexname{Index}
\def\figurename{Figure}
\def\tablename{Table}
\def\abstractname{A B S T R A C T}
\lineskip 1\p@
\normallineskip 1\p@
\def\baselinestretch{1}
\frenchspacing
\def\cropwidth{297mm}
\def\cropheight{210mm}
\def\cropleft{0mm}
\def\croptop{0mm}
\newdimen\@frontmatterwidth

\def\@overtitleskipwithtype{100\p@}
\def\@overtitleskipwithouttype{85\p@}

\def\@overhistoryskip{\z@}
\def\@undertitleskip{\z@}
\def\@overkeywordskip{12\p@}
\def\@overabstractskip{8\p@}
\def\@overcaptionskip{8\p@}
\def\@preabstractskip{24\p@ \@plus 6\p@ \@minus 3\p@}
\def\@belowfmskip{18\p@}
\def\@bibliosize{\small}
\def\@historysize{\small}
\def\@keywordsize{\small}
\def\@overaddressskip{2pt}
\def\@titlesize{\fontsize{14pt}{14pt}\bfseries}
\def\@bibitemsep{0.3\@bls \@plus 0.1\@bls \@minus 0.1\@bls}
\def\@authorsize{\large}
\def\@keywordheading{{\it Keywords: \ }}
\def\@addressstyle{\small\itshape}
\def\@captionsize{\small}
\def\@tablecaptionsize{\@captionsize}
\def\@figurecaptionsize{\@captionsize}
\def\@tablesize{\small}
\def\@keywordwidth{.8\textwidth}
\def\@abstractwidth{.8\textwidth}
\def\@fignumfont#1{#1}
\def\@secnumfont#1{\upshape}
\def\@catchlinesize{\footnotesize}
\def\@runheadsize{\normalsize}
\def\@narrowdisplaywidth{259pt}
\font\tencmi=cmmib10 scaled 1200
\def\@Colofonheight{1cm}
\def\@abstractsize{\fontsize{10pt}{10pt}\selectfont}
\def\@articletypesize{\fontsize{\@xiiipt}{13pt}\selectfont}

\newif\if@has@article@info
\newif\if@has@article@history
\newif\if@has@received
\newif\if@has@first@revised
\newif\if@has@second@revised
\newif\if@has@third@revised
\newif\if@has@accepted
\newif\if@has@published@online
\newif\if@has@abstract
\newif\if@has@keywords
\newif\if@has@info@or@abstract

\@has@article@infofalse
\@has@article@historyfalse
\@has@receivedfalse
\@has@first@revisedfalse
\@has@second@revisedfalse
\@has@third@revisedfalse
\@has@acceptedfalse
\@has@published@onlinefalse
\@has@abstractfalse
\@has@keywordsfalse
\@has@info@or@abstractfalse

\def\first@prefix{}

\def\receiveddate#1{%
	\global\@has@receivedtrue%
	\global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
	\gdef\@receiveddate{#1}}
\def\reviseddate#1{%
	\global\@has@first@revisedtrue%
	\global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
	\gdef\@first@reviseddate{#1}}
%
\def\secondreviseddate#1{%
  \global\@has@second@revisedtrue%
  \global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
  \gdef\first@prefix{First\ }%
  \gdef\@second@reviseddate{#1}}
%
\def\thirdreviseddate#1{%
  \global\@has@third@revisedtrue%
  \global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
  \gdef\first@prefix{First\ }%
  \gdef\@third@reviseddate{#1}}
%
\def\accepteddate#1{%
	\global\@has@acceptedtrue%
	\global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
	\gdef\@accepteddate{#1}}
\def\onlinedate#1{%
	\global\@has@published@onlinetrue%
	\global\@has@article@historytrue%
    \global\@has@article@infotrue%
    \global\@has@info@or@abstracttrue%
	\gdef\@onlinedate{#1}}

\def\normalsize{\@setfontsize\normalsize\@xiipt{14.5}%
\def\@ptsize{2}
\abovedisplayskip 12\p@ \@plus 2\p@ \@minus 2\p@%
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ \@plus 2\p@%
\belowdisplayshortskip 3.5\p@ \@plus 2\p@ \@minus 2\p@
\let\@listi\@listI}
\def\small{\@setfontsize\small\@xipt{13.6}%
\abovedisplayskip 11\p@ \@plus3\p@ minus6\p@
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip  \z@ \@plus3\p@
\belowdisplayshortskip  6.5\p@ \@plus3.5\p@ minus3\p@
\def\@listi{\leftmargin\leftmargini
 \parsep 4.5\p@ \@plus2\p@ minus\p@ \itemsep \parsep
            \topsep 9\p@ \@plus3\p@ minus5\p@}}
\let\footnotesize=\small
\let\@xviiipt\@xviipt
\def\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
\def\tiny{\@setfontsize\tiny\@vipt{7}}
\def\large{\@setfontsize\large\@xivpt{18}}
\def\Large{\@setfontsize\Large\@xviipt{22}}
\def\LARGE{\@setfontsize\LARGE\@xxpt{22}}
\def\huge{\@setfontsize\huge\@xxvpt{27}}
\let\Huge=\huge
\normalsize                                % Choose the normalsize font.
\newdimen\@bls                              % Several dimensions are
\@bls=\baselineskip                         % expressed in terms of this.
\if@twoside                 % Values for two-sided printing:
   \oddsidemargin   20\p@    %   Left margin on odd-numbered pages.
   \evensidemargin  20\p@    %   Left margin on even-numbered pages.
   \marginparwidth  10\p@    %   \@Width of marginal notes.
\else                       % Values for one-sided printing:
   \oddsidemargin   20\p@    %   Left margin on odd-numbered pages.
   \evensidemargin  20\p@    %   Left margin on even-numbered pages.
   \marginparwidth   2pc
\fi
\marginparsep 20\p@          % Horizontal space between outer margin and
                            % marginal note
\topmargin \z@           %    Nominal distance from top of page to top of
                         %    box containing running head.
\headheight  \z@         %    \@Height of box containing running head.
\headsep     \z@         %    Space between running head and text.
 \footskip 40\p@
\bigskipamount=\@bls \@plus 0.3\@bls \@minus 0.3\@bls % 1/1 line
\medskipamount=0.5\bigskipamount                  % 1/2 line
\smallskipamount=0.25\bigskipamount               % 1/4 line
\@tempdima=651pt \advance\@tempdima -\topskip \@tempcnta=\@tempdima
\@tempdimb=\@blstr\@bls \@tempcntb=\@tempdimb
\divide\@tempcnta\@tempcntb
\textheight\@tempdimb \multiply\textheight\@tempcnta
\advance\textheight\topskip
\textwidth 33pc              % \@Width of text line.
                             % For two-column mode:
 \columnsep 2pc               %   Space between columns
\columnseprule \z@           %   \@Width of rule between columns.
 \footnotesep 8.4\p@
\skip\footins 12\p@ \@plus  8\p@          % Space between last line of text and
                                      % top of first footnote.
\floatsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between adjacent floats moved
                                         % to top or bottom of text page.
\textfloatsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between main text and floats
                                         % at top or bottom of page.
\intextsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between in-text figures and
                                         % text.
\dblfloatsep      8\p@ \@plus 4\p@ \@minus 4\p@ % Same as \floatsep for double-column
                                         % figures in two-column mode.
\dbltextfloatsep 12\p@ \@plus 4\p@ \@minus 4\p@ % \textfloatsep for double-column
                                         % floats.
\@fptop \z@ \@plus 1fil    % Stretch at top of float page/column. (Must be
                         % \z@ \@plus ...)
\@fpsep 8\p@ \@plus 2fil    % Space between floats on float page/column.
\@fpbot \z@ \@plus 1fil    % Stretch at bottom of float page/column. (Must be
                         % \z@ \@plus ... )
\@dblfptop \z@ \@plus 1fil % Stretch at top of float page. (Must be \z@ \@plus ...)
\@dblfpsep 8\p@ \@plus 2fil % Space between floats on float page.
\@dblfpbot \z@ \@plus 1fil % Stretch at bottom of float page. (Must be
                         % \z@ \@plus ... )
\marginparpush 5\p@       % Minimum vertical separation between two marginal
                         % notes.
\parskip 1pc \@plus 1\p@          % Extra vertical space between paragraphs.
\parindent \z@                     % Indentation of each paragraph.
\newskip\eqntopsep                    % Extra vertical space, in addition to
 \eqntopsep 12\p@ \@plus 2\p@ \@minus 2\p@ %\parskip, added above and below
\newdimen\eqnarraycolsep            % Half the space between columns
\eqnarraycolsep 1\p@                 % in an \eqnarray.
\@lowpenalty   51      % Produced by \nopagebreak[1] or \nolinebreak[1]
\@medpenalty  151      % Produced by \nopagebreak[2] or \nolinebreak[2]
\@highpenalty 301      % Produced by \nopagebreak[3] or \nolinebreak[3]
\@beginparpenalty -\@lowpenalty    % Before a list or paragraph environment.
\@endparpenalty   -\@lowpenalty    % After a list or paragraph environment.
\@itempenalty     -\@lowpenalty    % Between list items.

\def\half@em{\hskip 0.5em}
\def\lb@part{PART \thepart.\half@em}
  \def\lb@empty@part{PART \thepart}
\def\lb@section{\thesection \half@em}
  \def\lb@empty@section{\thesection}
\def\lb@subsection{\thesubsection \half@em}
  \def\lb@empty@subsection{\thesubsection}
\def\lb@subsubsection{\thesubsubsection \half@em}
  \def\lb@empty@subsubsection{\thesubsubsection}
\def\lb@paragraph{\theparagraph.\half@em}
  \def\lb@empty@paragraph{\theparagraph}
\def\lb@subparagraph{\thesubparagraph.\half@em}
  \def\lb@empty@subparagraph{\thesubparagraph}
\def\head@format#1#2{#2}
\def\head@style{\interlinepenalty\@M
  \hyphenpenalty\@M \exhyphenpenalty\@M
  \rightskip \z@ \@plus 0.5\hsize \relax
  }
\def\app@number#1{\setcounter{#1}{0}%
  \@addtoreset{#1}{section}%
  \@namedef{the#1}{\thesection.\arabic{#1}}}
\def\appendix{\@ifstar{\appendix@star}{\appendix@nostar}}
\def\appendix@nostar{%
  \def\lb@section{\appendixname\ \thesection.\half@em}
  \def\lb@empty@section{\appendixname\ \thesection}
  \setcounter{section}{0}\def\thesection{\Alph{section}}%
  \setcounter{subsection}{0}%
  \setcounter{subsubsection}{0}%
  \setcounter{paragraph}{0}%
  \app@number{equation}\app@number{figure}\app@number{table}}
\def\appendix@star{%
  \def\lb@section{\appendixname}\let\lb@empty@section\lb@section
  \setcounter{section}{0}\def\thesection{\Alph{section}}%
  \setcounter{subsection}{0}%
  \setcounter{subsubsection}{0}%
  \setcounter{paragraph}{0}%
  \app@number{equation}\app@number{figure}\app@number{table}}
\def\ack{\section*{Acknowledgements}%
  \addtocontents{toc}{\protect\vspace{6pt}}%
  \addcontentsline{toc}{section}{Acknowledgements}%
}
\@namedef{ack*}{\par\vskip 3.0ex \@plus 1.0ex \@minus 1.0ex}
\let\endack\par
\@namedef{endack*}{\par}
\newdimen\labelwidthi
\newdimen\labelwidthii
\newdimen\labelwidthiii
\newdimen\labelwidthiv
\def\normal@labelsep{0.5em}
\labelsep\normal@labelsep
\settowidth{\labelwidthi}{(iii)}
\settowidth{\labelwidthii}{(d)}
\settowidth{\labelwidthiii}{(iii)}
\settowidth{\labelwidthiv}{(M)}
\leftmargini\labelwidthi    \advance\leftmargini\labelsep
\leftmarginii\labelwidthii  \advance\leftmarginii\labelsep
\leftmarginiii\labelwidthiii \advance\leftmarginiii\labelsep
\leftmarginiv\labelwidthiv  \advance\leftmarginiv\labelsep
\def\setleftmargin#1#2{\settowidth{\@tempdima}{#2}\labelsep\normal@labelsep
  \csname labelwidth#1\endcsname\@tempdima
  \@tempdimb\@tempdima \advance\@tempdimb\labelsep
  \csname leftmargin#1\endcsname\@tempdimb}
\def\@listI{\leftmargin\leftmargini
  \labelwidth\labelwidthi \labelsep\normal@labelsep
  \topsep \z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listii{\leftmargin\leftmarginii
  \labelwidth\labelwidthii \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listiii{\leftmargin\leftmarginiii
  \labelwidth\labelwidthiii \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\def\@listiv{\leftmargin\leftmarginiv
  \labelwidth\labelwidthiv \labelsep\normal@labelsep
  \topsep\z@ \partopsep\z@ \parsep\z@ \itemsep\z@
  \listparindent 1em}
\let\@listi\@listI
\@listi
\def\left@label#1{{#1}\hss}
\def\right@label#1{\hss\llap{#1}}
\def\thick@label#1{\hspace\labelsep #1}
\newcount\@maxlistdepth
\@maxlistdepth=4
\def\labelitemi{$\bullet$}          \def\labelitemii{$\circ$}           \def\labelitemiii{$-$}
\def\labelenumi{(\theenumi)}        \def\theenumi{\arabic{enumi}}
\def\labelenumii{(\alph{enumii})}   \def\theenumii{\theenumi\alph{enumii}}
\def\labelenumiii{(\roman{enumiii})}\def\theenumiii{\theenumii.\roman{enumiii}}
\def\labelenumiv{(\Alph{enumiv})}   \def\theenumiv{\theenumiii.\Alph{enumiv}}
\def\enumerate{%
  \ifnum \@enumdepth >\@maxlistdepth
    \@toodeep
  \else
    \advance\@enumdepth \@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}%
       {\usecounter{\@enumctr}
       \let\makelabel=\right@label}
  \fi}
\def\itemize{%
  \ifnum \@itemdepth >\@maxlistdepth
    \@toodeep
  \else
    \advance\@itemdepth \@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
     \setleftmargin{i}{--}%
     \setleftmargin{ii}{$\cdot$}%
    \list{\csname\@itemitem\endcsname}%
       {\ifnum \@itemdepth=1
           \leftmargin=2em
        \else
           \leftmargin=1em
        \fi
        \let\makelabel\right@label}
  \fi}
\def\verse{\let\\=\@centercr
  \list{}{\itemsep\z@
  \itemindent \z@
  \listparindent\z@
  \rightmargin 1em
  \leftmargin \rightmargin}\item[]}
\let\endverse\endlist
\def\quotation{\list{}{\itemindent\z@
 \leftmargin 1em \rightmargin \z@
  \parsep \z@ \@plus 1pt}\item[]}
\let\endquotation=\endlist
\def\quote{\list{}{\itemindent\z@
   \leftmargin 1em \rightmargin \z@}%
\item[]}
\let\endquote=\endlist
\def\descriptionlabel#1{\hspace\labelsep \bfseries #1}
\def\description{\list{}{\labelwidth\z@
  \leftmargin 1em \itemindent-\leftmargin
  \let\makelabel\descriptionlabel}}
\let\enddescription\endlist
\def\@atfmtname{atlplain}
\ifx\fmtname\@atfmtname
 \def\neq{\not\nobreak\mkern -2mu =}%
 \let\ne\neq
\fi
\def\operatorname#1{\mathop{\mathrm{#1}}\nolimits}
\def\lefteqn#1{\hbox to\z@{$\displaystyle {#1}$\hss}}
\newskip\eqnbaselineskip % Standard interline spacing in an {eqnarray}
\jot=2\p@
\newskip\eqnlineskip     % Minimal space between the bottom of
                         % a line and the top of the next line.
\eqnbaselineskip=14\p@  \eqnlineskip=2\p@
\newdimen\mathindent
\if@TwoColumn
  \mathindent 0em
\else
  \mathindent 2em
\fi
\if@narrowdisplay
  \def\calcmathindent{%
    \mathindent\linewidth\relax
    \advance\mathindent-\@narrowdisplaywidth}
\else
  \let\calcmathindent\relax
\fi
\def\[{\relax\ifmmode\@badmath
  \else%\bgroup removed on request from BW (1993-05-17)
  \@beginparpenalty\predisplaypenalty
  \@endparpenalty\postdisplaypenalty
\calcmathindent
  \begin{trivlist}\@topsep \eqntopsep       % used by first \item
   \@topsepadd \eqntopsep                   % used by \@endparenv
  \item[]\leavevmode
   \hbox to\linewidth\bgroup$ \displaystyle
  \hskip\mathindent\bgroup\fi}
\def\]{\relax\ifmmode \egroup $\hfil \egroup
  \end{trivlist}% \egroup removed on request from BW (1993-05-17)
  \else \@badmath \fi}
\def\equation{\@beginparpenalty\predisplaypenalty
  \@endparpenalty\postdisplaypenalty
  \refstepcounter{equation}\trivlist
  \@topsep \eqntopsep                      % used by first \item
  \@topsepadd \eqntopsep                   % used by \@endparenv
\calcmathindent
  \item[]\leavevmode
   \hbox to\linewidth\bgroup $ \displaystyle \hskip\mathindent\bgroup}
\def\endequation{\egroup$\hfil \displaywidth\linewidth
  \@eqnnum\egroup \endtrivlist}
\def\eqnarray{%
  \par                                               %BW
  \noindent                                          %BW
  \baselineskip\eqnbaselineskip\lineskip\eqnlineskip %BW
  \lineskiplimit\eqnlineskip                         %BW
\calcmathindent
  \stepcounter{equation}%
  \let\@currentlabel=\theequation
  \global\@eqnswtrue
  \global\@eqcnt\z@
  \tabskip\mathindent
  \let\\=\@eqncr
  \abovedisplayskip\eqntopsep\ifvmode\advance\abovedisplayskip\partopsep\fi
  \belowdisplayskip\abovedisplayskip
  \belowdisplayshortskip\abovedisplayskip
  \abovedisplayshortskip\abovedisplayskip
  $$\halign to \displaywidth\bgroup\@eqnsel
    \pre@coli$\displaystyle\tabskip\z@{##}$\post@coli
    &\global\@eqcnt\@ne
    \pre@colii$\displaystyle{##}$\post@colii
    &\global\@eqcnt\tw@
    \pre@coliii $\displaystyle\tabskip\z@{##}$\post@coliii
    \tabskip\@centering&\llap{##}\tabskip\z@\cr
}
\def\endeqnarray{\@@eqncr\egroup
 \global\advance\c@equation\m@ne$$\global\@ignoretrue }
\def\pre@coli{\hskip\@centering}              \def\post@coli{}
\def\pre@colii{\hskip 2\eqnarraycolsep \hfil} \def\post@colii{\hfil}
\def\pre@coliii{\hskip 2\eqnarraycolsep}      \def\post@coliii{\hfil}
\arraycolsep 2\p@         % Half the space between columns in array environment.
\tabcolsep 6\p@           % idem in tabular environment.
\def\arraystretch{1.5}   % More vertical space in tables
\arrayrulewidth 0.4\p@    % \@Width of rules and space between adjacent
\doublerulesep 2\p@       % rules in any of these two environments.
\newdimen\rulepreskip \newdimen\rulepostskip
\rulepreskip=4\p@      \rulepostskip=6\p@
\tabbingsep \labelsep   % Space used by the \' command.  (See LaTeX{} manual.)
\skip\@mpfootins = 6\p@ \@plus 2\p@   % Space between last line of text and
                                  % top of first footnote.
\fboxsep = 7\p@    % Space left between box and text by \fbox and \framebox.
\fboxrule = 0.4\p@ % \@Width of rules in box made by \fbox and \framebox.
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\if@seceqn
 \@addtoreset{equation}{section}
 \def\theequation{\arabic{section}.\arabic{equation}}
\else
  \def\theequation{\arabic{equation}}
\fi
\def\thesection      {\arabic{section}}
\def\thesubsection   {\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph    {\thesubsubsection.\arabic{paragraph}}
\def\thesubparagraph {\theparagraph.\arabic{subparagraph}}
\@addtoreset{section}{part} % reset section numbers at beginning of part

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\RequirePackage{latexsym}
\def\qed{\relax\ifmmode\hskip2em \Box\else\unskip\nobreak\hskip1em $\Box$\fi}
\def\proof@headerfont{\upshape\bfseries}
%\if@amsthm
%  \RequirePackage{amsthm}%
%  \def\th@plain{%
%    \thm@preskip\parskip
%    \thm@postskip\z@
%    \itshape % body font
%  }
%  \def\th@definition{%
%    \thm@preskip\parskip
%    \thm@postskip\z@
%    \normalfont % body font
%  }
%  \def\th@remark{%
%    \thm@headfont{\itshape}%
%    \normalfont % body font
%    \thm@preskip\parskip \divide\thm@preskip\tw@
%    \thm@postskip\z@
%  }
%\else\let\theoremstyle\@gobble\fi
%\newenvironment{pf}%
%  {\par\addvspace{\@bls \@plus 0.5\@bls \@minus 0.1\@bls}\noindent
%   {\bfseries\Elproofname}\enspace\ignorespaces}%
%  {\par\addvspace{\@bls \@plus 0.5\@bls \@minus 0.1\@bls}}
%\def\Elproofname{PROOF.}
%\@namedef{pf*}#1{\par\begingroup\def\Elproofname{#1}\pf\endgroup\ignorespaces}
%\expandafter\let\csname endpf*\endcsname=\endpf
%\theoremstyle{plain}
%\if@secthm
%    \newtheorem{theorem}{Theorem}[section]
%    \newtheorem{proposition}{Proposition}[section]
%    \newtheorem{corollary}{Corollary}[section]
%    \newtheorem{lemma}{Lemma}[section]
%    \newtheorem{claim}{Claim}[section]
%    \newtheorem{axiom}{Axiom}[section]
%    \newtheorem{conjecture}{Conjecture}[section]
%    \newtheorem{fact}{Fact}[section]
%    \newtheorem{hypothesis}{Hypothesis}[section]
%    \newtheorem{assumtion}{Assumption}[section]
%    \newtheorem{criterion}{Criterion}[section]
%    \theoremstyle{definition}
%    \newtheorem{definition}{Definition}[section]
%    \newtheorem{example}{Example}[section]
%    \newtheorem{remark}{Remark}[section]
%    \newtheorem{notation}{Notation}[section]
%    \newtheorem{problem}{Problem}[section]
%    \newtheorem{principle}{Principle}[section]
%
%    \@addtoreset{theorem}{section}
%    \@addtoreset{proposition}{section}
%    \@addtoreset{corollary}{section}
%    \@addtoreset{lemma}{section}
%    \@addtoreset{claim}{section}
%    \@addtoreset{axiom}{section}
%    \@addtoreset{conjecture}{section}
%    \@addtoreset{fact}{section}
%    \@addtoreset{hypothesis}{section}
%    \@addtoreset{assumtion}{section}
%    \@addtoreset{criterion}{section}
%    \@addtoreset{definition}{section}
%    \@addtoreset{example}{section}
%    \@addtoreset{remark}{section}
%    \@addtoreset{notation}{section}
%    \@addtoreset{problem}{section}
%    \@addtoreset{principle}{section}
%
%\else
%    \newtheorem{theorem}{Theorem}
%    \newtheorem{proposition}{Proposition}
%    \newtheorem{corollary}{Corollary}
%    \newtheorem{lemma}{Lemma}
%    \newtheorem{claim}{Claim}
%    \newtheorem{axiom}{Axiom}
%    \newtheorem{conjecture}{Conjecture}
%    \newtheorem{fact}{Fact}
%    \newtheorem{hypothesis}{Hypothesis}
%    \newtheorem{assumtion}{Assumption}
%    \newtheorem{criterion}{Criterion}
%    \theoremstyle{definition}
%    \newtheorem{definition}{Definition}
%    \newtheorem{example}{Example}
%    \newtheorem{remark}{Remark}
%    \newtheorem{notation}{Notation}
%    \newtheorem{problem}{Problem}
%    \newtheorem{principle}{Principle}
%\fi
\newtheorem{algorithm1}{Algorithm}
\long\def\@makealgocaption#1#2{\vskip 2ex \small
  \hbox to \hsize{\parbox[t]{\hsize}{{\bfseries #1.} #2}}}
\newcounter{algorithm}
\def\thealgorithm{\@arabic\c@algorithm}
\def\fps@algorithm{tbp}
\def\ftype@algorithm{4}
\def\ext@algorithm{lof}
\def\fnum@algorithm{Algorithm \thealgorithm}
\def\algorithm{\let\@makecaption\@makealgocaption\@float{algorithm}}
\def\endalgorithm{\end@float}
\newtheorem{note}{Note}
\newtheorem{summ}{Summary}
\newtheorem{case}{Case}
\def\@pnumwidth{2.55em}
\def\@tocrmarg{2.55em \@plus 5em}
\def\@dotsep{-2.5}
\setcounter{tocdepth}{2}
\newcommand\listoffigures{%
    \section*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename
      \@mkboth{%
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\def\tableofcontents{%
  \begin{small}
  \leftline {{\bfseries \contentsname\/}}
  \setcounter{secnumdepth}{4}%
  \setcounter{tocdepth}{2}%
  {\@starttoc{toc}}%
\end{small}
}
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\def\@dotsep{2000}
\def\thebibliography{\section*{\refname}%
  \@thebibliography}
\let\endthebibliography=\endlist
\def\@thebibliography#1{\@bibliosize
  \list{\@biblabel{\arabic{enumiv}}}{\settowidth\labelwidth{\@biblabel{#1}}
  \if@nameyear
    \labelwidth\z@ \labelsep\z@ \leftmargin\parindent
    \itemindent-\parindent
  \else
    \labelsep 3\p@ \itemindent\z@
    \leftmargin\labelwidth \advance\leftmargin\labelsep
\fi
\itemsep\@bibitemsep
    \usecounter{enumiv}\let\p@enumiv\@empty
    \def\theenumiv{\arabic{enumiv}}}%
    \tolerance\@M
   \hyphenpenalty\@M
     \hbadness5000 \sfcode`\.=1000\relax}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\if@nameyear
  \def\@biblabel#1{}
\else
 \def\@biblabel#1{[#1]\hskip \z@ \@plus 1filll}
\fi
\let\make@bb@error\relax
\def\@mkbberr{\def\bibitem{\ClassError{isecure}%
    {Bibitem after note}%
    {You are using a bibitem after a note in a subbibitems
      environment;\MessageBreak
      note should the last item in a subbibitems environment}}}
\def\@itemnote{\make@bb@error\item[]}
\def\mk@noitemnote{\ifx\@tempa\note \let\note\@noitemnote \fi}
\def\@noitemnote{\let\note\@itemnote}
\AtBeginDocument{%
  \let\nopeek@bibitem\@bibitem
  \let\nopeek@lbibitem\@lbibitem
  \def\@bibitem#1{\let\note\@itemnote\nopeek@bibitem{#1}%
    \futurelet\@tempa\mk@noitemnote}
  \def\@lbibitem[#1]#2{\let\note\@itemnote\nopeek@lbibitem[#1]{#2}%
    \futurelet\@tempa\mk@noitemnote}
}
\newif\if@natbibloaded\@natbibloadedfalse
\AtBeginDocument{\@ifpackageloaded{natbib}{\@natbibloadedtrue}{}}
\newenvironment{subbibitems}[1]{%
  \if@natbibloaded\def\bib@ctr{NAT@ctr}\else\def\bib@ctr{enumiv}\fi
  \if@filesw {\let \protect \noexpand \immediate \write \@auxout
    {\string \nocollapse@cites}%
    \global\let\nocollapse@cites\relax}\fi
  \def\@itemslabel{#1}%
  \stepcounter{\bib@ctr}%
  \edef\main@bibnum{\the\value{\bib@ctr}}%
  \setcounter{\bib@ctr}{0}%
  \def\thebib@ctr{\main@bibnum\alph{\bib@ctr}}%
  \if@natbibloaded
    \def\bibitem{\@ifnextchar [{\@lbibitem }{\global \NAT@stdbsttrue
        \stepcounter {\bib@ctr}\@lbibitem [\thebib@ctr]}}%
  \else
    \def\@bibitem##1{\stepcounter{\bib@ctr}\@lbibitem[\thebib@ctr]{##1}}%
  \fi
  \let\make@bb@error\@mkbberr
}{%
  \if@filesw {%
    \if@natbibloaded{%
      \let\protect\noexpand
      \immediate\write\@auxout {\string\bibcite{\@itemslabel}%
        {{\main@bibnum a--\alph {\bib@ctr}}{}{{}}{{}}}}%
      \immediate\write\@auxout {\string\bibcite{\@itemslabel :s}%
        {{\main@bibnum}{}{{}}{{}}}}%
    }\else{%
      \let\protect\noexpand
      \immediate\write\@auxout {\string\bibcite{\@itemslabel}%
        {\main@bibnum a--\alph{\bib@ctr}}}%
      \immediate\write\@auxout {\string\bibcite{\@itemslabel :s}%
        {\main@bibnum}}%
    }\fi
  }\fi
  \setcounter{\bib@ctr}{\main@bibnum}%
}
\let\nocollapse@citex\@citex
\newcount\@tempcntc
\def\collapse@citex[#1]#2{\if@filesw\immediate\write\@auxout{\string\citation{#2}}\fi
 \@tempcnta\z@\@tempcntb\m@ne\def\@citea{}\@cite{\@for\@citeb:=#2\do
  {\edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
   \@ifundefined
   {b@\@citeb}{\@citeo\@tempcntb\m@ne\@citea\def\@citea{,}{\bfseries ?}\@warning
   {Citation `\@citeb' on page \thepage \space undefined}}%
  {\setbox\z@\hbox{\global\@tempcntc0\csname b@\@citeb\endcsname\relax}%
   \ifnum\@tempcntc=\z@ \@citeo\@tempcntb\m@ne
    \@citea\def\@citea{,}\hbox{\csname b@\@citeb\endcsname}%
   \else
    \advance\@tempcntb\@ne
    \ifnum\@tempcntb=\@tempcntc
    \else\advance\@tempcntb\m@ne\@citeo
    \@tempcnta\@tempcntc\@tempcntb\@tempcntc\fi\fi}}\@citeo}{#1}}
\let\@citex\collapse@citex
\def\nocollapse@cites{%
  \@ifpackageloaded{hyperref}{}{\global\let\@citex\nocollapse@citex}%
  \global\let\nocollapse@cites\relax}
\def\@citeo{\ifnum\@tempcnta>\@tempcntb\else\@citea\def\@citea{,}%
 \ifnum\@tempcnta=\@tempcntb\the\@tempcnta\else
  {\advance\@tempcnta\@ne\ifnum\@tempcnta=\@tempcntb \else \def\@citea{--}\fi
   \advance\@tempcnta\m@ne\the\@tempcnta\@citea\the\@tempcntb}\fi\fi}
\@namedef{cv*}{\section*{Curriculum Vitae}\cv}
 \def\cv{\hangindent=7pc \hangafter=-12 \parskip\bigskipamount \small}
\def\footnote{\@ifnextchar[{\@xfootnote}{\refstepcounter
   {\@mpfn}\protected@xdef\@thefnmark{\thempfn}\@footnotemark\@footnotetext}}
\def\footnotemark{\@ifnextchar[{\@xfootnotemark
    }{\refstepcounter{footnote}\xdef\@thefnmark{\thefootnote}\@footnotemark}}
\def\footnoterule{\kern-3\p@
  \hrule \@width 3pc               % The \hrule has default \@height of 0.4pt.
  \kern 2.6\p@}
\def\thempfootnote{\alph{mpfootnote}}
\def\mpfootnotemark{%
  \@ifnextchar[{\@xmpfootnotemark}{\stepcounter{mpfootnote}%
  \begingroup
    \let\protect\noexpand
    \xdef\@thefnmark{\thempfootnote}%
  \endgroup
  \@footnotemark}}
\def\@xmpfootnotemark[#1]{%
  \begingroup
    \c@mpfootnote #1\relax
    \let\protect\noexpand
    \xdef\@thefnmark{\thempfootnote}%
  \endgroup
  \@footnotemark}
\def\@mpmakefnmark{\,\hbox{$^{\mathrm{\@thefnmark}}$}}
\long\def\@mpmakefntext#1{\noindent
                     \hbox{$^{\mathrm{\@thefnmark}}$} #1}
\def\@iiiminipage#1#2[#3]#4{%
  \leavevmode
  \@pboxswfalse
  \setlength\@tempdima{#4}%
  \def\@mpargs{{#1}{#2}[#3]{#4}}%
  \setbox\@tempboxa\vbox\bgroup
    \color@begingroup
      \hsize\@tempdima
      \textwidth\hsize \columnwidth\hsize
      \@parboxrestore
      \def\@mpfn{mpfootnote}\def\thempfn{\thempfootnote}\c@mpfootnote\z@
      \let\@footnotetext\@mpfootnotetext
      \let\@makefntext\@mpmakefntext
      \let\@makefnmark\@mpmakefnmark
      \let\@listdepth\@mplistdepth \@mplistdepth\z@
      \@minipagerestore\global\@minipagetrue %% \global added 24 May 89
      \everypar{\global\@minipagefalse\everypar{}}}
\def\fn@presym{}
\long\def\@makefntext#1{\noindent\hbox to 1em
  {$^{\fn@presym\mathrm{\@thefnmark}}$\hss}#1}
\def\@makefnmark{\,\hbox{$^{\fn@presym\mathrm{\@thefnmark}}$}\,}
\def\patched@end@dblfloat{%
  \if@twocolumn
      \par\vskip\z@skip      %% \par\vskip\z@ added 15 Dec 87
      \global\@minipagefalse
      \outer@nobreak
    \egroup                  %% end of vbox
    \color@endbox
    \ifnum\@floatpenalty <\z@
      \@largefloatcheck
      \@cons\@dbldeferlist\@currbox
    \fi
    \ifnum \@floatpenalty =-\@Mii \@Esphack\fi
  \else
    \end@float
  \fi
}
\setcounter{topnumber}{5}
\def\topfraction{0.99}
\def\textfraction{0.05}
\def\floatpagefraction{0.9}
\setcounter{bottomnumber}{5}
\def\bottomfraction{0.99}
\setcounter{totalnumber}{10}
\def\dbltopfraction{0.99}
\def\dblfloatpagefraction{0.8}
\setcounter{dbltopnumber}{5}

% The following lines commented for making the table caption like the firgure caption
%\long\def\@maketablecaption#1#2{\@tablecaptionsize
%    \global \@minipagefalse
%    \hbox to \hsize{\parbox[t]{\hsize}{#1 \\ #2}}}

\long\def\@maketablecaption#1#2{\@tablecaptionsize
  \vskip \@overcaptionskip
  \setbox\@tempboxa\hbox{#1. #2}
  \ifdim \wd\@tempboxa >\hsize              % IF longer than one line THEN
    \unhbox\@tempboxa\par                   %   set as justified paragraph
  \else                                     % ELSE
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%   center single line.
  \fi
  \vspace{2mm}}


\long\def\@makefigurecaption#1#2{\@figurecaptionsize
  \vskip \@overcaptionskip
  \setbox\@tempboxa\hbox{#1. #2}
  \ifdim \wd\@tempboxa >\hsize              % IF longer than one line THEN
    \unhbox\@tempboxa\par                   %   set as justified paragraph
  \else                                     % ELSE
    \global \@minipagefalse
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%   center single line.
  \fi}
\def\@makecaption{\@makefigurecaption}
\def\conttablecaption{\par \begingroup \@parboxrestore \normalsize
  \@makecaption{\fnum@table\,---\,continued}{}\par
  \vskip-1pc \endgroup}
\def\contfigurecaption{\vskip-1pc \par \begingroup \@parboxrestore
  \@captionsize
  \@makecaption{\fnum@figure\,---\,continued}{}\par
  \endgroup}
\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\textbf{\figurename~\thefigure}}
\def\figure{%
 \let\@makecaption\@makefigurecaption
  \let\contcaption\contfigurecaption \@float{figure}}
\def\endfigure{\end@float}
\@namedef{figure*}{%
 \let\@makecaption\@makefigurecaption
  \let\contcaption\contfigurecaption \@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}
\newcounter{table}
\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\textbf{\tablename~\thetable}}
\let\old@floatboxreset\@floatboxreset
\def\table{%
\let\@makecaption\@maketablecaption
\def\@floatboxreset{%
 \old@floatboxreset
 \@tablesize
}%
  \let\footnoterule\relax
  \let\contcaption\conttablecaption \@float{table}}
\def\endtable{\end@float}
\@namedef{table*}{%
\let\@makecaption\@maketablecaption
\def\@floatboxreset{%
 \old@floatboxreset
 \@tablesize
}%
  \let\footnoterule\relax
  \let\contcaption\conttablecaption \@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}
\newtoks\t@glob@notes             % List of all notes
\newtoks\t@loc@notes              % List of notes for one element
\newcount\note@cnt                % Number of notes per element
\newtoks\corauth@text
\newtoks\email@text
\newtoks\url@text
\newcounter{corauth}
\newcounter{author}               % Author counter
\newcount\n@author                % Total number of authors
\def\n@author@{1}                  % idem, read from .aux file
\newcounter{collab}               % Collaboration counter
\newcount\n@collab                % Total number of collaborations
\def\n@collab@{}                  % idem, read from .aux file
\newcounter{address}              % Address counter
\def\theHaddress{\arabic{address}}% for hyperref
\newdimen\sv@mathsurround         % Dimen register to save \mathsurround
\newcount\sv@hyphenpenalty        % Count register to save \hyphenpenalty
\newcount\prev@elem \prev@elem=0  % Variables to keep track of
\newcount\cur@elem  \cur@elem=0   % types of elements that are processed
\chardef\e@title=1
\chardef\e@subtitle=1
\chardef\e@author=2
\chardef\e@collab=3
\chardef\e@address=4
\newif\if@newelem                 % Switch to new type of element?
\newif\if@firstauthor             % First author or collaboration?
\newif\if@preface                 % If preface: omit history and abstract
\newbox\fm@box                    % Box for collected front matter
\newdimen\fm@size                 % Total height of \fm@box
\newbox\t@abstract                % Box for abstract
\newbox\t@keyword                 % Box for keyword abstract
 \let\report@elt\@gobble
\def\add@tok#1#2{\global#1\expandafter{\the#1#2}}
\def\add@xtok#1#2{\begingroup
  \no@harm
  \xdef\@act{\global\noexpand#1{\the#1#2}}\@act
\endgroup}
\def\beg@elem{\global\t@loc@notes={}\global\note@cnt\z@}
\def\@xnamedef#1{\expandafter\xdef\csname #1\endcsname}
\def\no@harm{%
  \let\\=\relax  \let\rm\relax
  \let\ss=\relax \let\ae=\relax \let\oe=\relax
  \let\AE=\relax \let\OE=\relax
  \let\o=\relax  \let\O=\relax
  \let\i=\relax  \let\j=\relax
  \let\aa=\relax \let\AA=\relax
  \let\l=\relax  \let\L=\relax
  \let\d=\relax  \let\b=\relax \let\c=\relax
  \let\bar=\relax
  \def\protect{\noexpand\protect\noexpand}}
\def\proc@elem#1#2{\begingroup
    \no@harm                             % make a few instructions harmless
    \let\thanksref\@gobble               % remove \thanksref from element
    \let\corauthref\@gobble
    \@xnamedef{@#1}{#2}%                 % and store as \@#1
    \let\thanksref\add@thanksref
    \let\corauthref\add@thanksref
    \setbox\@tempboxa\hbox{#2}%
  \endgroup
  \prev@elem=\cur@elem                   % keep track of type of previous
  \cur@elem=\csname e@#1\endcsname       % and current element
}
\def\add@thanksref#1{\global\advance\note@cnt\@ne
  \ifnum\note@cnt>\@ne \add@xtok\t@loc@notes{\note@sep}\fi
  \add@tok\t@loc@notes{\ref{#1}}}
\def\note@sep{,}
\def\thanks{\@ifnextchar[{\@tempswatrue
  \thanks@optarg}{\@tempswafalse\thanks@optarg[]}}
\def\thanks@optarg[#1]#2{\refstepcounter{footnote}%
  \if@tempswa\label{#1}\else\relax\fi
  \add@tok\t@glob@notes{\footnotetext}%
  \add@xtok\t@glob@notes{[\the\c@footnote]}%
  \add@tok\t@glob@notes{{#2}}%
  \ignorespaces}
\def\corauth{\@ifnextchar[{\@tempswatrue
  \corauth@optarg}{\@tempswafalse\corauth@optarg[]}}
\def\corauth@optarg[#1]#2{\refstepcounter{corauth}%
  \if@tempswa\label{#1}\else\relax\fi
  \add@tok\corauth@text{\footnotetext}%
  \add@xtok\corauth@text{[\the\c@corauth]}%
  \add@tok\corauth@text{{#2}}\ignorespaces}

\def\TitleComment{\@ifnextchar[{\@tempswatrue
  \TitleComment@optarg}{\@tempswafalse\TitleComment@optarg[]}}

\def\TitleComment@optarg[#1]#2{%
  \if@tempswa\label{#1}\else\relax\fi
  \add@tok\corauth@text{\footnotetext}%
  \add@xtok\corauth@text{[\the\c@corauth]}%
  \add@tok\corauth@text{{#2}}\ignorespaces}

\def\noemail{\mbox{}}
\def\ead#1{
    \def\inner{#1}
    \if\inner\noemail
        \@eadNo{#1}
    \else
        \@ead{#1}
    \fi
}

\newcommand\@eadNo[1][email]{%
  \add@ead{#1}{}%
}

\newcommand\@ead[1][email]{%
  \add@eadcomma{#1}%
  \expandafter\ifcase\csname has@ead@#1\endcsname
  \expandafter\global\expandafter\chardef
    \csname has@ead@#1\endcsname=1\relax\else
  \expandafter\global\expandafter\chardef
    \csname has@ead@#1\endcsname=2\relax\fi
  \add@ead{#1}%
}
\def\add@ead#1#2{%
  \expandafter\add@tok\csname #1@text\endcsname{\texttt{#2}\ead@au}%
%  \expandafter\add@xtok\csname #1@text\endcsname{{\@author}}%
  \ignorespaces
}
\def\add@eadcomma#1{%
  \expandafter\ifcase\csname has@ead@#1\endcsname\else
  \expandafter\add@tok\csname #1@text\endcsname{, }\fi
}
\let\@ead@au\@empty
\def\ead@newau{\ifx\@ead@au\@empty\else\ead@addau\@ead@au
  \let\@ead@au\@empty\fi}
\let\ead@endau\ead@newau
\def\ead@addau#1{ (#1)}
\def\ead@au#1{\def\@ead@au{#1}}
\def\email@name{Email address}
\def\emails@name{Email addresses}
\chardef\has@ead@email=0
\def\url@name{URL}
\def\urls@name{URLs}
\chardef\has@ead@url=0
\let\real@refstepcounter\refstepcounter
\def\footnote{\@ifnextchar[{\@xfootnote}{\real@refstepcounter
   {\@mpfn}\protected@xdef\@thefnmark{\thempfn}\@footnotemark\@footnotetext}}
\def\footnotemark{\@ifnextchar[{\@xfootnotemark
    }{\real@refstepcounter{footnote}\xdef\@thefnmark{\thefootnote}\@footnotemark}}
\def\footnoterule{\kern-3\p@
  \hrule \@width 3pc               % The \hrule has default \@height of 0.4pt.
  \kern 2.6\p@}
 \let\report@elt\@gobble
%\newenvironment{NoHyper}{}{}
\def\preface{\@prefacetrue}
\def\endfrontmatter{%
  \ifx\@runauthor\relax
   \global\let\@runauthor\@runningauthor
  \fi
  \global\n@author=\c@author
  \global\n@collab=\c@collab \@writecount
  \global\@topnum\z@
  \thispagestyle{copyright}%            % Format rest of front matter:
  \if@preface \else                     % IF not preface THEN
  \vskip \@overhistoryskip
  \history@fmt                          % print history (received, ...)
  \newcount\c@sv@footnote
  \global\c@sv@footnote=\c@footnote     % save current footnote number

\if@has@info@or@abstract
    \begin{TabularX}[b]{cc}%
    \if@has@article@info
        \hline
        \parbox[t]{5cm}{
        \parbox[t]{4.5cm}{
          \vspace*{-7pt}
          \fontsize{8pt}{8pt}
          {\bfseries A R T I C L E ~ I N F O.}
          \vspace*{1pt}
          \@keywordsize
          \justify@off                         % Keywords are not justified.
          \if@has@article@history
              \vskip 0.45\@bls    % half a line of space below
              \hrule
              \vskip 0.3\@bls    % half a line of space below
              \parskip\z@
              {\it Article history:}
              {
                \font\sixroman=cmr6
                \font\sixbf=cmbx6
                \sixroman
                  \if@has@received
                    \\ {\sixbf Received:} \@receiveddate
                  \fi
                  \if@has@first@revised
                    \\{\sixbf \first@prefix Revised:} \@first@reviseddate
                  \fi
                  \if@has@second@revised
                    \\{\sixbf Last Revised:} \@second@reviseddate
                  \fi
                  \if@has@third@revised
                    \\{\sixbf Third Revised:} \@third@reviseddate
                  \fi
                  \if@has@accepted
                    \\{\sixbf Accepted:} \@accepteddate
                  \fi
                  \if@has@published@online
                    \\{\sixbf Published Online:} \@onlinedate
                  \fi
              }
          \fi
          \vskip 0.3\@bls    % half a line of space below
          \hrule
          \vskip 0.3\@bls    % half a line of space below
          \if@has@keywords                       % IF keywords THEN
            %\vskip \@overkeywordskip
          \@keywordheading\\
            \unhbox\t@keyword                   % Keyword abstract, if any
            \vskip 0.3\@bls    % half a line of space below
            \hrule
          \fi                                   % FI
          }
        }
    \fi
    &
    \if@has@abstract
        \begin{minipage}[t]{12cm}
            \vskip -8\p@
            \unhbox\t@abstract                  % print abstract, if any
            \vspace{-0.5\baselineskip}
            \\
            \vbox{\hrule height 0.4pt width 12cm}
        \end{minipage}
    \fi
    \end{TabularX}%
\fi


  \if@hastoc                       % IF keywords THEN
    \vskip \@overkeywordskip
    \ltx@tableofcontents                   % Keyword abstract, if any
  \fi                                   % FI
  \vskip 10\p@
%  \hrule height 0.4\p@                 % rule below abstract/keywords
  \dedicated@fmt                        % print dedication
  \vskip \@belowfmskip                  % Vertical space below frontmatter
\fi                                   % FI
  \close@fm                             % Close front matter material.
   \output@glob@notes  % Put notes at bottom of 1st page
  \endNoHyper
  \global\c@footnote=\c@sv@footnote     % restore footnote number
  \global\@prefacefalse
  \global\leftskip\z@                   % Restore the normal values of
  \global\@rightskip\z@                 % \leftskip,
  \global\rightskip\@rightskip          % \rightskip and
  \global\mathsurround\sv@mathsurround  % \mathsurround.
  \let\title\relax       \let\author\relax
  \let\collab\relax      \let\address\relax
  \let\frontmatter\relax \let\endfrontmatter\relax
  \let\@maketitle\relax  \let\@@maketitle\relax
  \normal@text
}
\let\maketitle\relax
\newdimen\t@xtheight
\t@xtheight\textheight \advance\t@xtheight-\splittopskip
\let\nolinenumbers\relax
\def\open@fm{\global\setbox\fm@box=\vbox\bgroup\nolinenumbers
  \hsize=\@frontmatterwidth                 % Front matter is page-wide by default
  \centering                                % and centered
  \sv@hyphenpenalty\hyphenpenalty           % (save \hyphenpenalty)
  \hyphenpenalty\@M}                        % and not hyphenated
\def\close@fm{\egroup                       % close \vbox (\fm@box)
  \fm@size=\dp\fm@box \advance\fm@size by \ht\fm@box
  \@whiledim\fm@size>\t@xtheight \do{%
    \global\setbox\@tempboxa=\vsplit\fm@box to \t@xtheight
    \unvbox\@tempboxa \newpage
    \fm@size=\dp\fm@box \advance\fm@size by \ht\fm@box}
  \if@TwoColumn
    \emergencystretch=1pc \twocolumn[\unvbox\fm@box]
  \else
    \unvbox\fm@box
  \fi}
\def\output@glob@notes{\bgroup
  \the\t@glob@notes
  \egroup}
\def\justify@off{\let\\=\@normalcr
  \leftskip\z@ \@rightskip\@flushglue \rightskip\@rightskip}
\def\justify@on{\let\\=\@normalcr
  \leftskip\z@ \@rightskip\z@ \rightskip\@rightskip}
\def\normal@text{\global\let\\=\@normalcr
  \global\leftskip\z@ \global\@rightskip\z@ \global\rightskip\@rightskip
  \global\parfillskip\@flushglue}
\def\@writecount{\write\@mainaux{\string\global
  \string\@namedef{n@author@}{\the\n@author}}%
  \write\@mainaux{\string\global\string
  \@namedef{n@collab@}{\the\n@collab}}}
\def\title#1{%
\raggedright
  \beg@elem
  \title@note@fmt                      % formatting instruction
  \add@tok\t@glob@notes                % for \thanks commands
    {\title@note@fmt}%
  \proc@elem{title}{#1}%
  \def\title@notes{\the\t@loc@notes}%  % store the notes of the title,
  \title@fmt{\@title}{\title@notes}%   % print the title
  \ignorespaces}
\def\subtitle#1{%
  \beg@elem
  \proc@elem{subtitle}{#1}%
  \def\title@notes{\the\t@loc@notes}%  % store the notes of the title,
  \subtitle@fmt{\@subtitle}{\title@notes}% print the title
  \ignorespaces}
\newdimen \@logoheight \@logoheight 5pc
\def\@Lhook{\vrule \@height \@logoheight \@width \z@ \vrule \@height 10\p@ \@width 0.2\p@ \vrule \@height 0.2\p@ \@width 10pt}
\def\@Rhook{\vrule \@height 0.2\p@ \@width 10\p@ \vrule \@height 10\p@ \@width 0.2\p@ \vrule \@height \@logoheight \@width \z@}
\def\subtitle@fmt#1#2{%               % No vertical space above sub-title
  {\@titlesize #1\,\hbox{$^{#2}$}}\par}
\def\title@note@fmt{\def\thefootnote{\fnstar{footnote}}}
\def\author{\@ifnextchar[{\author@optarg}{\author@optarg[]}}
\def\author@optarg[#1]#2{\stepcounter{author}%
  \beg@elem
  \add@tok\email@text{\ead@newau}%
  \add@tok\url@text{\ead@newau}%
  \@for\@tempa:=#1\do{\expandafter\add@thanksref\expandafter{\@tempa}}%
  \report@elt{author}\proc@elem{author}{#2}%
  \ifnum0\n@collab@=\z@ \runningauthor@fmt \fi
  \author@fmt{\the\c@author}{\the\t@loc@notes}{\@author}%
}
\def\runningauthor@fmt{%
  \begingroup\no@harm
    \if@firstauthor
      \ifnum0\n@author@ > 2
        \global\edef\@runningauthor{\@author\ et al.}%
      \else
        \global\let\@runningauthor\@author%
      \fi
    \else % \c@author > 1
      \ifnum0\n@author@ = 2
        \global\edef\@runningauthor{\@runningauthor, \noexpand\@author}%
      \fi
    \fi
  \endgroup
}
\def\author@fmt#1#2#3{\@newelemtrue
  \if@firstauthor
  \first@author \global\@firstauthorfalse \fi
  \ifnum\prev@elem=\e@author \global\@newelemfalse \fi
  \if@newelem \author@fmt@init \fi
  \edef\@tempb{#2}\ifx\@tempb\@empty
    \hbox{{\author@font #3}}\else
    \hbox{{\author@font #3}\,$^{\mathrm{#2}}$}%
  \fi}
\def\copyright@note{
    \let\thefootnote\relax
    \footnotetext[0]{
        \raggedright{%
            \\
            \vspace{-0.8\baselineskip}
            \@fullcopyright
        }
    }
}

\def\first@author{\author@note@fmt
  \corauth@mark@fmt
  \add@tok\t@glob@notes
  {\output@corauth@text
    \output@ead@text{email}%
    \output@ead@text{url}%
    \author@note@fmt
    \copyright@note}%
}%
\def\author@fmt@init{%
  \par
  \vskip 8\p@ \@plus 4\p@ \@minus 2\p@
  \@authorsize
  \leavevmode}                        % Vertical space above author list
 \def\and{\unskip~and~}
\def\collab{\@ifstar{\collab@arg}{\collab@arg}}
\let\collaboration=\collab
\def\collab@arg#1{\stepcounter{collab}%
  \if@firstauthor \first@author \global\@firstauthorfalse \fi
  \gdef\@runningauthor{#1}%
  \beg@elem
  \proc@elem{collab}{#1}%
  \collab@fmt{\the\c@collab}{\the\t@loc@notes}{\@collab}%
  \ignorespaces}
\def\collab@fmt#1#2#3{\@newelemtrue
  \ifnum\prev@elem=\e@collab \global\@newelemfalse \fi
  \if@newelem \collab@fmt@init \fi
  \par                                 % Start new paragraph
  \edef\@tempb{#2}\ifx\@tempb\@empty
    {\@authorsize #3}\else
    {\@authorsize #3\,$^{\mathrm{#2}}$}%
  \fi}
\def\collab@fmt@init{\vskip 1em}       % Vertical space above list
\def\author@note@fmt{\setcounter{footnote}{0}%
  \def\thefootnote{\xarabic{footnote}}}
\let\collab@note@fmt=\author@note@fmt
\def\corauth@mark@fmt{\def\thecorauth{\astsymbol{corauth}}}
\def\output@corauth@text{\def\thefootnote{\astsymbol{footnote}}%
  \the\corauth@text}
\def\output@ead@text#1{%
  \expandafter\add@tok\csname #1@text\endcsname{\ead@endau}%
  \expandafter\ifcase\csname has@ead@#1\endcsname\else
  {\let\thefootnote\relax
    \footnotetext[0]{\raggedright{%
        \vspace{-0.8\baselineskip}
        \\
        \expandafter\ifcase\csname has@ead@#1\endcsname\or
        \csname #1@name\endcsname\else
        \csname #1s@name\endcsname\fi
        : }%
        \expandafter\the\csname #1@text\endcsname
        }}%
    \fi}
\def\xarabic#1{%
  \expandafter\expandafter\expandafter\ifnum\expandafter\the\@nameuse{c@#1}<0
  *\else\arabic{#1} \fi}
\def\xalph#1{%
  \expandafter\expandafter\expandafter\ifnum\expandafter\the\@nameuse{c@#1}<0
  *\else\alph{#1} \fi}
\def\xfnsymbol#1{%
  \expandafter\expandafter\expandafter\ifnum\expandafter\the\@nameuse{c@#1}<0
  *\else\fnsymbol{#1} \fi}
\def\address{\@ifstar{\address@star}%
  {\@ifnextchar[{\address@optarg}{\address@noptarg}}}
\def\address@optarg[#1]#2{\real@refstepcounter{address}%
  \beg@elem
  \report@elt{address}\proc@elem{address}{#2}%
  \address@fmt{\c@address}{\the\t@loc@notes}{\@address}{#1}%
  \if@Elproofing\else\label{#1}\fi
  \ignorespaces}
\def\address@noptarg#1{\real@refstepcounter{address}%
  \beg@elem
  \proc@elem{address}{#1}%
  \address@fmt{\z@}{\the\t@loc@notes}{\@address}{\theaddress}%
  \ignorespaces}
\def\address@star#1{%
  \beg@elem
  \proc@elem{address}{#1}%
  \address@fmt{\m@ne}{\the\t@loc@notes}{\@address}{*}%
  \ignorespaces}
\def\theaddress{\arabic{address}}
\def\address@fmt#1#2#3#4{\@newelemtrue
  \if@Elproofing\def\@eltag{#4}\else\def\@eltag{\theaddress}\fi
  \ifnum\prev@elem=\e@address \@newelemfalse \fi
  \if@newelem \address@fmt@init \fi
  \noindent \bgroup \@addressstyle
  \ifnum#1=\z@
    #3\,$^{\mathrm{#2}}$\space%
  \else
    \ifnum#1=\m@ne
      $^{\phantom{\mathrm{\@eltag}}}$\space #3\,$^{\mathrm{#2}}$%
    \else
      $^{\mathrm{\@eltag}}\space$#3\,$^{\mathrm{#2}}$%
    \fi
  \fi
  \par \egroup}
\def\address@fmt@init{%
  \par                                % Start new paragraph
   \vskip 6\p@ \@plus 3\p@ \@minus 1.5pt}
\def\abstract{\@ifnextchar[{\@abstract}{\@abstract[]}}

\def\@abstract[#1]{%
  \global\@has@abstracttrue
  \global\@has@info@or@abstracttrue%
  \hyphenpenalty\sv@hyphenpenalty     % restore \hyphenpenalty
  \global\setbox\t@abstract=\hbox\bgroup
  \leftskip\z@
  \@rightskip\z@ \rightskip\@rightskip \parfillskip\@flushglue
  \normalsize
  \parindent 1em                      % \parindent in abstract
  \vbox{
  \noindent {\bfseries\abstractname}  % caption `Abstract' (bold)
  \vskip -0.3\@bls    % half a line of space below
  \vbox{\hrule height 0.4pt width 12cm}%
  }
\noindent\ignorespaces
}

\def\@shortcopyright{}

\def\endabstract{%
 \newline%
 \mbox{}\hfill{\@shortcopyright}%
 \egroup %\egroup works much the same as the right curly bracket
}

\def\keyword{%
  \global\@has@keywordstrue             % Implies rules are to be printed
  \global\@has@article@infotrue%
  \global\@has@info@or@abstracttrue%
  \hyphenpenalty\sv@hyphenpenalty      % restore \hyphenpenalty
  \global\setbox\t@keyword=\hbox\bgroup
  \@keywordsize
  \justify@off                         % Keywords are not justified.
  \parskip\z@
  }

\def\endkeyword{\par\egroup}
\def\runtitle#1{\gdef\@runtitle{#1}}
\def\runauthor#1{\gdef\@runauthor{#1}}
\let\@runauthor\relax
\let\@runtitle\relax
\let\@runningauthor\relax
\def\RUNDATE{}
\def\RUNJNL{}
\def\RUNART{}
\def\journal#1{\gdef\@journal{#1}}     \def\@journal{ISeCure}
\def\volume#1{\gdef\@volume{#1}}       \def\@volume{0}
\def\issue#1{\gdef\@issue{#1}}         \def\@issue{0}
\def\journalsite#1{\gdef\@journalsite{#1}} \def\@journalsite{}
\def\publishmonth#1{\gdef\@publishmonth{#1}}
\def\@publishmonth{\ifcase\month\or%
  January\or February\or March\or April\or May\or June\or%
  July\or August\or September\or October\or November\or December\fi}
\def\publishyear#1{\gdef\@publishyear{#1}}   %\def\@publishyear{\the\year}
\def\copyrightyear#1{\gdef\@copyrightyear{#1}}
\newcount\@pubyear
%\newrgbcolor{pagenumbercol}{0.0 0.0 1.0}
\@pubyear=\number\year
\def\@shortenyear#1#2#3#4\\{\global\def\@shortyear{#3#4}}
\expandafter\@shortenyear\the\@pubyear\\
\newcounter{firstpage}
\newcounter{lastpage}
\let\ESpagenumber\arabic
\def\firstpage#1{\def\@tempa{#1}\ifx\@tempa\@empty\else
  \setcounter{firstpage}{#1}%
  \global\c@page=#1 \ignorespaces\fi}
\setcounter{firstpage}{1}
\let\realpageref\pageref
\setcounter{lastpage}{0}
\def\lastpage#1{\def\@tempa{#1}\ifx\@tempa\@empty\else
  \setcounter{lastpage}{#1}\ignorespaces\fi
  }
\AtEndDocument{%
   \clearpage
   \addtocounter{page}{-1}%
   \immediate\write\@auxout{%
   \string\global\string\c@lastpage=\the\c@page}%
   \addtocounter{page}{1}%
}
\def\date#1{\gdef\@date{#1}}                  \def\@date{\today}
\def\aid#1{}
\def\ssdi#1#2{}
\def\received#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@received{#1}\fi}
  \def\@received{\relax}
\def\revised#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@revised{#1}\fi}
  \def\@revised{\relax}
\def\accepted#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@accepted{#1}\fi}
  \def\@accepted{\relax}
\def\communicated#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@communicated{#1}\fi}
  \def\@communicated{\relax}
\def\dedicated#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@dedicated{#1}\fi}
  \def\@dedicated{\relax}
\def\presented#1{\def\@tempa{#1}\ifx\@tempa\@empty\else\gdef\@presented{#1}\fi}
  \def\@presented{\relax}
\def\articletype#1{\gdef\@articletype{#1}}
  \@ifundefined{@articletype}{\def\@articletype{}}{}
\def\received@prefix{Received~}
\def\revised@prefix{; revised~}
\def\accepted@prefix{; accepted~}
\def\communicated@prefix{; communicated~by~}
\def\history@prefix{}
\def\received@postfix{}
\def\revised@postfix{}
\def\accepted@postfix{}
\def\communicated@postfix{}
\def\history@postfix{}
\def\empty@data{\relax}
\def\history@fmt{%
  \bgroup
  \@historysize
  \vskip 6\p@ \@plus 2\p@ \@minus 1\p@         % Vertical space above history
  \ifx\@received\empty@data \else       % If there is no \received,
                                        % do not print anything
    \leavevmode
    \history@prefix
    \received@prefix\@received \received@postfix%
    \ifx\@revised\empty@data \else
      \revised@prefix\@revised \revised@postfix%
    \fi
    \ifx\@accepted\empty@data \else
      \accepted@prefix\@accepted \accepted@postfix%
    \fi
    \ifx\@communicated\empty@data \else
      \communicated@prefix\@communicated \communicated@postfix%
    \fi
    \history@postfix
  \fi
  \par \egroup}
\def\dedicated@fmt{%
  \ifx\@dedicated\empty@data \else
    \vskip 4\p@ \@plus 3\p@
    \normalsize\it\centering \@dedicated
    \fi}
\def\@alph#1{\ifcase#1\or a\or b\or c\or d\or e\or f\or g\or h\or i\or
j\or k\or \ensuremath{\ell}\or m\or n\or o\or p\or q\or r\or s\or t\or
u\or v\or w\or x\or y\or z\or aa\or ab\or ac\or ad\or ae\or af\or ag\or
ah\or ai\or aj\or ak\or a\ensuremath{\ell}\or am\or an\or ao\or ap\or
aq\or ar\or as\or at\or au\or av\or aw\or ay\or az\or ba\or bb\or bc\or
bd\or be\or bf\or bg\or bh\or bi\or bj\or bk\or b\ensuremath{\ell}\or
bm\or bn\or bo\or bp\or bq\or br\or bs\or bt\or bu\or bw\or bx\or by\or
bz\or ca\or cb\or cc\or cd\or ce\or cf\or cg\or ch\or ci\or cj\or ck\or
c\ensuremath{\ell}\or cm\or cn\or co\or cp\or cq\or cr\or cs\or ct\or
cu\or cw\or cx\or cy\or cz\or da\or db\or dc\or dd\or de\or df\or dg\or
dh\or di\or dj\or dk\or d\ensuremath{\ell}\or dm\or dn\or do\or dp\or
dq\or dr\or ds\or dt\or du\or dw\or dx\or dy\or dz\or ea\or eb\or ec\or
ed\or ee\or ef\or eg\or eh\or ei\or ej\or ek\or e\ensuremath{\ell}\or
em\or en\or eo\or ep\or eq\or er\or es\or et\or eu\or ew\or ex\or ey\or
ez\else\@ctrerr\fi}
\def\fnstar#1{\@fnstar{\@nameuse{c@#1}}}
\def\@fnstar#1{\ifcase#1\or
    \hbox{$\star$}\or
    \hbox{$\star\star$}\or
    \hbox{$\star\star\star$}\or
    \hbox{$\star\star\star\star$}\or
    \hbox{$\star\star\star\star\star$}\or
    \hbox{$\star\star\star\star\star\star$}
  \else
    \@ctrerr
  \fi
  \relax}
\def\astsymbol#1{\@astsymbol{\@nameuse{c@#1}}}
\def\@astsymbol#1{\ifcase#1\or
    \hbox{$\ast$}\or
    \hbox{$\ast\ast$}\or
    \hbox{$\ast\ast\ast$}\or
    \hbox{$\ast\ast\ast\ast$}\or
    \hbox{$\ast\ast\ast\ast\ast$}\or
    \hbox{$\ast\ast\ast\ast\ast\ast$}%
  \else
    \@ctrerr
  \fi
  \relax}
\mark{{}{}}   % Initializes TeX's marks

\def\Page@Range{%
    \ifnum\c@firstpage=\c@lastpage
        (p.~\thefirstpage%
    \else
        (pp.~\thefirstpage--\thelastpage%
    \fi
    )%
}

\def\ps@plain{\let\@mkboth\@gobbletwo
    \def\@oddhead{%
        \put(484, 5){\includegraphics[width=80pt]{StyleFigs/\ColorDir/Blue.pdf}}
        %\put(0,10){\makebox[17cm][c]{\bfseries \footnotesize \@publishmonth\space\@publishyear, Volumeeee \@volume, Number \@issue~\Page@Range}}
        \put(487,10){\rmfamily \color{white} \thepage}
        \includegraphics[width=17cm]{StyleFigs/\ColorDir/RulerOdd.pdf}
    }%
    \def\@evenhead{
        \put(-80, 5){\includegraphics[width=80pt]{StyleFigs/\ColorDir/Blue.pdf}}
        \put(-80,10){\makebox[76pt][r]{\rmfamily \color{white} \thepage}}
        \ifx\@runauthor\empty
            \def\HeadSepMark{}
        \else
            \def\HeadSepMark{~---~}
        \fi
        \ifx\@runtitle\empty
            \relax
        \else
            \put(0,10){\makebox[17cm][c]{\bfseries \footnotesize\@runtitle\HeadSepMark\@runauthor}}
        \fi
        \includegraphics[width=17cm]{}
    }%
    \def\@oddfoot{%
        \includegraphics[width=17cm]{}
    }%
    \def\@evenfoot{
        \includegraphics[width=17cm]{}
    }%
}
\def\@fullcopyright{EN3160 - Image Processing and Machine Vision}
\def\@jou@vol@pag{\@journal\ \@volume\ (\the\@pubyear)\ \@pagerange}
\def\sectionmark#1{}
\def\subsectionmark#1{}
\let\@j@v@p\@jou@vol@pag    % long journal title appears in reprint line
\let\@@j@v@p\@jou@vol@pag   % long journal title appears in running headline
\def\sectionmark#1{}
\def\subsectionmark#1{}
\def\ps@copyright{\let\@mkboth\@gobbletwo

%################# FIRST PAGE HEADER AND FOOTER ####################
\newlength{\required@width}
\setlength{\required@width}{\textwidth+0.8em}

\def\@oddhead{
\hspace*{-0.8em}
\begin{TabularX}{lr}
    \multicolumn{2}{c}{
        \vbox{\vspace{6.5mm}
        {\color{BLACK}\hrule width \required@width height 3pt depth 0pt}}
    }\\
    \begin{TabularX}{c}
        \parbox[b]{8cm}{\includegraphics[width=8cm]{StyleFigs/\ColorDir/ENTC_UOM.png}}
    \end{TabularX}
    &
    \begin{TabularX}{r} {
      \font\tenrm=cmr12
      \tenrm
      Image Processing and Machine Vision\space\Page@Range} \\
      {\font\twelvebf=cmbx12 \twelvebf \@journalsite}
    \end{TabularX} \\
    \multicolumn{2}{b{\textwidth}}{\vbox{\color{BLACK} \hrule width \required@width height 2pt depth 0pt}}
\end{TabularX}
  }%
  \let\@evenhead\@oddhead
  \def\@oddfoot{%
      \includegraphics[width=10cm]{}
  }%
  \let\@evenfoot\@oddfoot
}
\let\ps@noissn\ps@empty
\let\ps@headings\ps@plain
\def\today{\number\day\space\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\year}
\def\nuc#1#2{\relax\ifmmode{}^{#1}{\protect\text{#2}}\else${}^{#1}$#2\fi}
\def\itnuc#1#2{\setbox\@tempboxa=\hbox{\scriptsize\it #1}
  \def\@tempa{{}^{\box\@tempboxa}\!\protect\text{\it #2}}\relax
  \ifmmode \@tempa \else $\@tempa$\fi}
\let\old@vec\vec % save old definition of \vec
\def\pol#1{\old@vec{#1}}
\def\half{{\textstyle\frac{1}{2}}}
\def\threehalf{{\textstyle\frac{3}{2}}}
\def\quart{{\textstyle\frac{1}{4}}}
\if@symbold\else\def\d{\,\mathrm{d}}\fi
\def\e{\mathop{\mathrm{e}}\nolimits}
\def\int{\intop}
\def\oint{\ointop}
\newbox\slashbox \setbox\slashbox=\hbox{$/$}
\newbox\Slashbox \setbox\Slashbox=\hbox{\large$/$}
\def\pFMslash#1{\setbox\@tempboxa=\hbox{$#1$}
  \@tempdima=0.5\wd\slashbox \advance\@tempdima 0.5\wd\@tempboxa
  \copy\slashbox \kern-\@tempdima \box\@tempboxa}
\def\pFMSlash#1{\setbox\@tempboxa=\hbox{$#1$}
  \@tempdima=0.5\wd\Slashbox \advance\@tempdima 0.5\wd\@tempboxa
  \copy\Slashbox \kern-\@tempdima \box\@tempboxa}
\def\FMslash{\protect\pFMslash}
\def\FMSlash{\protect\pFMSlash}
  \def\Cset{\mathbb{C}}
  \def\Hset{\mathbb{H}}
  \def\Nset{\mathbb{N}}
  \def\Qset{\mathbb{Q}}
  \def\Rset{\mathbb{R}}
  \def\Zset{\mathbb{Z}}
\if@TwoColumn
  \adjdemerits=100
  \linepenalty=100
  \doublehyphendemerits=5000        % experimental (1993-12-14)
  \emergencystretch=1.6pc
  \spaceskip=0.3em \@plus 0.17em \@minus 0.12em
\fi
\@frontmatterwidth\textwidth
\ps@headings                                % 'headings' page style
\pagenumbering{arabic}                      % Arabic page numbers
\def\thepage{\@pagenumprefix\ESpagenumber{page}}  % preceded by \@pagenumprefix
\let\baselinestretch\@blstr
\InputIfFileExists{\@shortjid.cfg}{}{}


\let\@journal\@save@journal
\let\@company\@save@company
\let\@shortjid\@save@shortjid

\def\@xiiipt{12}
\def\@titlesize{\fontsize{14pt}{14pt}\bfseries}
\def\@bibitemsep{\z@ \@plus 0.5\p@ \@minus 0.5\p@}
\def\captionwidth{.8\linewidth}
\def\@abstractsize{\fontsize{10pt}{10pt}\selectfont}
\def\@articletypesize{\fontsize{\@xiiipt}{13pt}\selectfont}
 \def\normalsize{\@setfontsize\normalsize\@xpt{12}%
\def\@ptsize{0}
\abovedisplayskip 5.75\p@ \@plus 2\p@ \@minus 2\p@%
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ \@plus 2\p@%
\belowdisplayshortskip 3.5\p@ \@plus 2\p@ \@minus 2\p@
\let\@listi\@listI}
\def\small{\@setfontsize\small\@viiipt{10}%
\abovedisplayskip 7\p@ \@plus 2\p@ \@minus 4\p@%
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip \z@ \@plus 1\p@%
\belowdisplayshortskip 3\p@ \@plus 1\p@ \@minus 2\p@
\def\@listi{\topsep 0.5\@bls \parsep\z@ \itemsep\parsep}}
\let\footnotesize=\small
\let\@xviiipt\@xviipt
\def\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
\def\tiny{\@setfontsize\tiny\@vipt{7}}
\def\large{\@setfontsize\large\@xiiipt{14}}
\def\Large{\@setfontsize\Large\@xviipt{20}}
\def\LARGE{\@setfontsize\LARGE\@xviiipt{22}}
\def\huge{\@setfontsize\huge\@xxpt{22}}
\def\Huge{\@setfontsize\Huge\@xxvpt{27}}
\normalsize                                % Choose the normalsize font.
\if@twoside
   \oddsidemargin    \z@
   \evensidemargin   \z@
   \marginparwidth  10\p@
\else
   \oddsidemargin    \z@
   \evensidemargin   \z@
   \marginparwidth   2pc
\fi
\marginparsep 20\p@          % Horizontal space between outer margin and
                            % marginal note
\topmargin \z@           %    Nominal distance from top of page to top of
                         %    box containing running head.
\headheight 10\p@         %
 \headsep    11\p@  %
 \footskip 23\p@
\bigskipamount=\@bls \@plus 0.3\@bls \@minus 0.3\@bls % 1/1 line
\medskipamount=0.5\bigskipamount                  % 1/2 line
\smallskipamount=0.25\bigskipamount               % 1/4 line
 \textheight 51\baselineskip
\advance\textheight\topskip  % excl. running head and foot).
 \textwidth 39pc              % \@Width of text line.
                             % For two-column mode:
 \columnsep 2pc               %   Space between columns
\columnseprule \z@           %   \@Width of rule between columns.
 \footnotesep 6.65\p@
\skip\footins 12\p@ \@plus  8\p@          % Space between last line of text and
                                      % top of first footnote.
\floatsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between adjacent floats moved
                                         % to top or bottom of text page.
\textfloatsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between main text and floats
                                         % at top or bottom of page.
\intextsep 8\p@ \@plus 4\p@ \@minus 2\p@ % Space between in-text figures and
                                         % text.
\dblfloatsep      8\p@ \@plus 4\p@ \@minus 4\p@ % Same as \floatsep for double-column
                                         % figures in two-column mode.
\dbltextfloatsep 12\p@ \@plus 4\p@ \@minus 4\p@ % \textfloatsep for double-column
                                         % floats.
\@fptop \z@ \@plus 1fil    % Stretch at top of float page/column. (Must be
                         % \z@ \@plus ...)
\@fpsep 8\p@ \@plus 2fil    % Space between floats on float page/column.
\@fpbot \z@ \@plus 1fil    % Stretch at bottom of float page/column. (Must be
                         % \z@ \@plus ... )
\@dblfptop \z@ \@plus 1fil % Stretch at top of float page. (Must be \z@ \@plus ...)
\@dblfpsep 8\p@ \@plus 2fil % Space between floats on float page.
\@dblfpbot \z@ \@plus 1fil % Stretch at bottom of float page. (Must be
                         % \z@ \@plus ... )
\marginparpush 5\p@       % Minimum vertical separation between two marginal
                         % notes.
\parskip \z@
\parindent 1em
 \eqntopsep 8\p@ \@plus 2\p@ \@minus 2\p@ %\parskip, added above and below
\eqnarraycolsep 1\p@                 % in an \eqnarray.
\def\part{\@startsection{part}{1}{\z@}{2\@bls \@plus
  0.4\@bls}{\@bls}{\normalsize}}
\def\partmark#1{}
%\def\@sect#1#2#3#4#5#6[#7]#8{%
%  \ifnum #2>\c@secnumdepth
%    \let\@svsec\@empty
%  \else
%    \refstepcounter{#1}\def\@tempa{#8}%
%    \ifx\@tempa\@empty
%      \edef\@svsec{\csname lb@empty@#1\endcsname}%
%    \else
%      \edef\@svsec{\csname lb@#1\endcsname}\fi\fi
%  \@tempskipa #5\relax
%  \ifdim \@tempskipa>\z@
%    \begingroup #6\relax
%\noindent \hskip #3\relax{\@secnumfont{#2}\@svsec}%
%  {\head@style\head@format{#2}{#8}\par}%
%    \endgroup
%    \csname #1mark\endcsname{#7}%
%    \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else
%     \protect\numberline{\csname the#1\endcsname}\fi #7}%
%  \else
%    \def\@svsechd{#6\hskip #3\relax {\@secnumfont{#2}\@svsec }%
%      {\head@style\head@format{#2}{#8}}\csname #1mark\endcsname{#7}%
%    \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth \else
%       \protect\numberline{\csname the#1\endcsname}\fi #7}}%
%  \fi
%  \@xsect{#5}}
%\def\@ssect#1#2#3#4#5#6{%
%  \@tempskipa #4\relax
%  \ifdim \@tempskipa>\z@
%    \begingroup
%      #5{%
%        \@hangfrom{\hskip #2}%
%          \interlinepenalty \@M \head@format{#1}{#6}\@@par}%
%    \endgroup
%  \else
%    \def\@svsechd{#5{\hskip #2\relax \head@format{#1}{#6}}}%
%  \fi
%  \@xsect{#4}}
%\def\@startsection#1#2#3#4#5#6{%
%  \if@noskipsec \leavevmode \fi
%  \par
%  \@tempskipa #4\relax
%  \@afterindenttrue
%  \ifdim \@tempskipa <\z@
%    \@tempskipa -\@tempskipa \@afterindentfalse
%  \fi
%  \if@nobreak
%    \everypar{}%
%  \else
%    \addpenalty\@secpenalty\addvspace\@tempskipa
%  \fi
%  \@ifstar
%    {\@ssect{#2}{#3}{#4}{#5}{#6}}%
%    {\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}

\@maxlistdepth=2
%\def\labelitemi{--}
%\def\labelitemii{$\cdot$}
%\def\labelenumi{(\theenumi)}        \def\theenumi{\roman{enumi}}
%\def\labelenumii{(\theenumii)}      \def\theenumii{\alph{enumii}}
 \tabcolsep 1\p@           % idem in tabular environment.
\def\@arrayclassz{\ifcase \@lastchclass \@acolampacol \or \@ampacol \or
  \or \or \@addamp \or \@acolampacol \or \@firstampfalse \@acol \fi
  \edef\@preamble{\@preamble
  \ifcase \@chnum
    \hfil$\relax\displaystyle\@sharp$\hfil \or
    $\relax\displaystyle\@sharp$\hfil      \or
    \hfil$\relax\displaystyle\@sharp$\fi}}
\newdimen\Arrayrulewidth \Arrayrulewidth=1.0\p@
\def\Hline{\noalign{\ifnum0=`}\fi\hrule \@height \Arrayrulewidth \futurelet
   \@tempa\@xhline}
\newdimen\crulepostskip \crulepostskip -7\p@
\def\ccline#1{%
  \noalign{\vskip\rulepreskip}%
  \do@ccline(#1,1-0)\cr
  \noalign{\vskip\crulepostskip}}
\def\do@ccline(#1-#2,#3){%
  \ifnum#1>#2\else\@cline[#1-#2]\do@ccline(#3,1-0)\fi
  \ignorespaces}
\newdimen\@Ldec \newdimen\@Rdec
\def\dec #1.#2 {\hbox to\@Ldec{\hss#1}\def\@tempb{#2}%
\ifx\@tempb\empty\hbox to\@Rdec{\hfill}\else\hbox to\@Rdec{.#2\hss}\fi}
\def\setdec #1.#2 {\relax
  \bgroup
    \setbox0\hbox{\kern1pt\relax#1}\setbox1\hbox{\kern1pt\relax#2}%
    \global\@Ldec\wd0\global\@Rdec\wd1
  \egroup}
\setdec 000.000 % default 3 digits on each side
\newdimen\@Lpmdec \newdimen\@Rpmdec
\def\pmdec #1.#2 {\hbox to\@Lpmdec{\hss#1}\def\@tempb{#2}%
\ifx\@tempb\empty\hbox to\@Rpmdec{\hfill}\else\hbox to\@Rpmdec{.#2\hss}\fi}
\def\setpmdec #1.#2 {\relax
  \bgroup
    \setbox0\hbox{\kern1pt\relax#1}\setbox1\hbox{\kern1pt\relax#2}%
    \global\@Lpmdec\wd0\global\@Rpmdec\wd1
  \egroup}
\setpmdec 0.0 % default 1 digit on each side
\def\add@thanksref#1{%
  \global\advance\note@cnt\@ne
  \def\@tempa{*}\def\@tempb{#1}%
  \ifx\@tempa\@tempb
    \ifnum\note@cnt>\@ne \add@xtok\t@loc@notes{\note@sep}\fi
    \add@tok\t@loc@notes{*}%
  \else
    \ifnum\note@cnt>\@ne \add@xtok\t@loc@notes{\note@sep}\fi
    \add@tok\t@loc@notes{%
    \if@Elproofing#1\else\ref{#1}\fi
    }%
  \fi}
\def\thanks@optarg[#1]#2{%
  \def\@tempa{*}\def\@tempb{#1}%
  \ifx\@tempa\@tempb
    \@tempcnta=\c@footnote \c@footnote=-1 \label{#1}\c@footnote=\@tempcnta
    \gdef\@corresp@note{\footnotetext[-1]{#2}}%
  \else
    \add@tok\t@glob@notes{\footnotetext}%
    \refstepcounter{footnote}%
    \if@Elproofing\else\if@tempswa\label{#1}\else\relax\fi\fi
    \add@xtok\t@glob@notes{[\the\c@footnote]}%
    \add@tok\t@glob@notes{{#2}}%
  \fi
  \ignorespaces}
\def\frontmatter{%
  \newif\if@hastoc\@hastocfalse
  \let\ltx@tableofcontents\tableofcontents
  \def\tableofcontents{\global\@hastoctrue}
  \NoHyper
  \let\@corresp@note\relax
  \global\t@glob@notes={}\global\c@author\z@
  \global\c@collab\z@ \global\c@address\z@
  \sv@mathsurround\mathsurround \m@th
  \global\n@author=0\n@author@\relax
  \global\n@collab=0\n@collab@\relax
  \global\advance\n@author\m@ne   % In comparisons later on we need
  \global\advance\n@collab\m@ne   % n@author-1 and n@collab-1
  \global\@firstauthortrue        % set to false by first \author or \collab
  \global\@has@abstractfalse       % Default:  no abstract or keywords
  \global\@has@keywordsfalse       % Default:  no abstract or keywords
  \global\@prefacefalse           %           not preface
  \ifnum\c@firstpage=\c@lastpage
    \gdef\@pagerange{\@pagenumprefix\ESpagenumber{firstpage}}
  \else
    \gdef\@pagerange{\@pagenumprefix
   \ESpagenumber{firstpage}--\@pagenumprefix\ESpagenumber{lastpage}}%
  \fi
  \@ifundefined{RIfM@}{}{\global\let\vec\@bfvec}%
  \open@fm \ignorespaces}
\t@xtheight\textheight \advance\t@xtheight-\splittopskip
\def\title@fmt#1#2{%
\@ifundefined{@runtitle}{\global\def\@runtitle{}}{}%
 \if@witharticletype
     \vspace*{\@overtitleskipwithtype}
     \hspace*{-0.57em}
      \@articletypesize                  % Size for article type
      \leavevmode\vphantom{Aye!}
      \textsc{\@articletype}
 \else
        \vspace*{\@overtitleskipwithouttype} % Vertical space above article type,
 \fi
  \vskip12\p@
  {\@titlesize \baselineskip=20pt #1\,\hbox{$^{#2}$}\par}%
  \vskip\@undertitleskip
  }
\def\address@fmt@init{%
  \par                                % Start new paragraph
  \vskip \@overaddressskip}               % Vertical space before addresses
\def\@bfvec#1{\boldsymbol{#1}}
\def\@pmbfvec#1{\pmb{#1}}
\mathchardef\Gamma="0000
\mathchardef\Delta="0001
\mathchardef\Theta="0002
\mathchardef\Lambda="0003
\mathchardef\Xi="0004
\mathchardef\Pi="0005
\mathchardef\Sigma="0006
\mathchardef\Upsilon="0007
\mathchardef\Phi="0008
\mathchardef\Psi="0009
\mathchardef\Omega="000A
\mathchardef\varGamma="0100
\mathchardef\varDelta="0101
\mathchardef\varTheta="0102
\mathchardef\varLambda="0103
\mathchardef\varXi="0104
\mathchardef\varPi="0105
\mathchardef\varSigma="0106
\mathchardef\varUpsilon="0107
\mathchardef\varPhi="0108
\mathchardef\varPsi="0109
\mathchardef\varOmega="010A
\if@TwoColumn
  \adjdemerits=100
  \linepenalty=100
  \doublehyphendemerits=5000        % experimental (1993-12-14)
  \emergencystretch=1.6pc
  \spaceskip=0.3em \@plus 0.17em \@minus 0.12em
\fi
\binoppenalty=300
\relpenalty=100
\clubpenalty=5000                % 'Club line'  at bottom of page.
\widowpenalty=2000               % 'Widow line' at top of page.
\displaywidowpenalty=1000        % Math display widow line.
\predisplaypenalty=150           % Breaking before a math display.
\postdisplaypenalty=50           % Breaking after a math display.
\hfuzz=1\p@
\hbadness=3000
\@frontmatterwidth\textwidth
\setlength{\columnsep}{0.39in}
\setlength{\textwidth}{6.69in}
\setlength{\textheight}{9.32in}
\setlength{\hoffset}{-0.2in}
\setlength{\voffset}{-0.5in}
\setlength{\headsep}{0.43in}
\setlength{\footskip}{0.68in}
\font\tenbf=cmb10 scaled 1200%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                 %
%   Section, Subsection, and Subsubsection definitions go here    %
%                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The following sets the depth of adding number to sections.
% Example: If set to 3, then sections, subsections, and subsubsections
% are numbered, but paragraph and subparagraphs aren't.

\setcounter{secnumdepth}{3}

\newcommand{\needspace}[1]{\begingroup\setlength{\dimen@}{#1}%
  \vskip\z@\@plus\dimen@ \penalty -100\vskip\z@\@plus -\dimen@
  \vskip\dimen@ \penalty 9999\vskip -\dimen@\endgroup}% from needspace.sty

\def\section{%
    \ifnum\thesection>0%    Draw a line over sections with sec-number>0
        \needspace{1\baselineskip}%
        \vskip 1.2\@bls%
        \hrule height 1pt%
    \fi
    \@startsection{section}{1}{\z@}{-1pt}{1pt}{\raggedright\normalfont\large\bfseries}%
}

\def\subsection{\@startsection{subsection}{2}{\z@}{-0.6\@bls}{1pt}{\raggedright\normalsize\bf}}

\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-0.6\@bls}{1pt}{\raggedright\normalsize\bf}}

\def\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {1pt}%
                                    {-1em}%
                                    {\normalfont\fontsize{9pt}{9pt}\bfseries}}

\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\itshape}}

\renewcommand{\footnoterule}{%
\kern -3pt % call this kerna
\hrule height 0.4pt width 0.75\columnwidth
\kern 2.6pt % call this kernb
}
\endinput
%%
%% End of file `isecure.cls'.
