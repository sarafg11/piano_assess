{% macro color() -%}
     \override NoteHead.color = #(rgb-color%{ next_color() %})
{%- endmacro %}
{% macro eps(scale, first_bar, last_bar) -%}
_\markup {
  \general-align #Y #DOWN {
    \epsfile #X #%{scale%} #"%{ eps_waveform(first_bar, last_bar, w=0.05*scale, h=0.35, left_border_shift = -0.05, right_border_shift=0) %}"
  }
}
{%- endmacro %}
#(set! paper-alist (cons '("my size" . (cons (* 8.55 in) (* 6 in))) paper-alist))
  
\paper {
  #(set-paper-size "my size")
}
\header {
  tagline = ""  % removed
}   

global = {
    \key c \major
    \time 4/4
 }

symbols =
        {
                \tempo 4=40
				\key c \major
				\time 2/4
                %{color()%}c'16 %{eps(103, 0, 3)%} %{color()%}e'16 %{color()%}f'16 %{color()%}g'16 %{color()%}a'16 %{color()%}g'16 %{color()%}f'16 %{color()%}e'16 
				\bar "|"
                %{color()%}d'16 %{color()%}f'16 %{color()%}g'16 %{color()%}a'16 %{color()%}b'16 %{color()%}a'16 %{color()%}g'16 %{color()%}f'16
				\bar "|"
				%{color()%}e'16 %{color()%}g'16 %{color()%}a'16 %{color()%}b'16 %{color()%}c''16 %{color()%}b'16 %{color()%}a'16 %{color()%}g'16
				\bar "|"
				%{color()%}f'16 %{color()%}a'16 %{color()%}b'16 %{color()%}c''16 %{color()%}d''16 %{color()%}c''16 %{color()%}b'16 %{color()%}a'16
				\bar"|."

        }
   
\score {
    \symbols

    \layout {
	indent = #0
        \context {
        \Score
         proportionalNotationDuration = #(ly:make-moment 1/10)
        }
    }
}
\version "2.18.2"  % necessary for upgrading to future LilyPond versions.
