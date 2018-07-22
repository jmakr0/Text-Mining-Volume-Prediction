create table authors
(
  author_id      integer not null
    constraint authors_pkey
    primary key,
  comment_author text
);

create unique index authors_c1_uindex
  on authors (author_id);

create table articles
(
  article_id  integer not null
    constraint articles_pkey
    primary key,
  article_url text
);

create unique index articles_article_id_uindex
  on articles (article_id);



create table comments
(
  article_id        integer   not null,
  author_id         integer   not null,
  comment_id        integer   not null
    constraint comments_comment_id_pk
    unique,
  comment_text      text,
  timestamp         timestamp not null,
  parent_comment_id integer,
  upvotes           integer
);

create index comments_article_id_index
  on comments (article_id);

create table guardianapi
(
  article_id           integer not null
    constraint guardianapi_pkey
    primary key,
  headline             text,
  commentclosedate     timestamp,
  standfirst           text,
  commentable          boolean,
  ispremoderated       boolean,
  lastmodified         timestamp,
  newspapereditiondate timestamp,
  legallysensitive     boolean,
  bodytext             text,
  sectionid            text,
  sectionname          text,
  webpublicationdate   timestamp,
  tags                 json
);

create unique index guardianapi_article_id_uindex
  on guardianapi (article_id);

create table categories
(
  id       serial not null,
  category text
);

create table doc2vec
(
  article_id integer             not null,
  vector     double precision [] not null,
  dimensions integer             not null,
  tag        text
);

create table guardianapierror
(
  article_id integer,
  reason     text
);

create view downloadview as
  SELECT a.article_id,
    a.article_url
   FROM ((articles a
     LEFT JOIN guardianapi g ON ((a.article_id = g.article_id)))
     LEFT JOIN guardianapierror ge ON ((a.article_id = ge.article_id)))
  WHERE ((g.article_id IS NULL) AND (ge.article_id IS NULL));

CREATE MATERIALIZED VIEW articlecommentcountview AS
  SELECT
    g.article_id,
    date_part('week' :: text, g.webpublicationdate) AS week,
    date_part('year' :: text, g.webpublicationdate) AS year,
    count(c.*)                                      AS comment_count
  FROM (guardianapi g
    LEFT JOIN comments c ON ((g.article_id = c.article_id)))
  GROUP BY g.article_id;

CREATE MATERIALIZED VIEW articlecountrankview AS
  SELECT
    ac.article_id,
    ac.week,
    ac.year,
    ac.comment_count,
    rank()
    OVER (
      PARTITION BY ac.week, ac.year
      ORDER BY ac.comment_count DESC ) AS rank
  FROM articlecommentcountview ac;

CREATE MATERIALIZED VIEW weeklytenpercentview AS
  SELECT
    ac.week,
    ac.year,
    ceil(((count(*)) :: numeric * 0.1)) AS ten_percent
  FROM articlecommentcountview ac
  GROUP BY ac.week, ac.year;

CREATE MATERIALIZED VIEW competitivearticlesview AS
  SELECT
    g2.article_id,
    g1.article_id                AS competitive_id,
    (abs((date_part('epoch' :: text, g1.webpublicationdate) - date_part('epoch' :: text, g2.webpublicationdate))) /
     (3600) :: double precision) AS diff
  FROM (guardianapi g1
    CROSS JOIN guardianapi g2)
  WHERE (
    (abs((date_part('epoch' :: text, g1.webpublicationdate) - date_part('epoch' :: text, g2.webpublicationdate))) <=
     (10800) :: double precision) AND (NOT (g1.article_id = g2.article_id)));

CREATE MATERIALIZED VIEW competitiveview AS
  SELECT
    ca.article_id,
    sum(competitive_score(ca.diff, d1.vector, d2.vector, d1.dimensions)) AS score
  FROM ((competitivearticlesview ca
    JOIN doc2vec d1 ON ((ca.article_id = d1.article_id)))
    JOIN doc2vec d2 ON ((ca.competitive_id = d2.article_id)))
  GROUP BY ca.article_id;

CREATE MATERIALIZED VIEW labelsview AS
  SELECT
    ga.article_id,
    ga.headline,
    sum(array_length(regexp_split_to_array(ga.headline, '\s' :: text), 1)) AS headline_word_count,
    ga.bodytext                                                            AS article,
    sum(array_length(regexp_split_to_array(ga.bodytext, '\s' :: text), 1)) AS article_word_count,
    c.id                                                                   AS category_id,
    date_part('epoch' :: text, ga.webpublicationdate)                      AS unix_timestamp,
    date_part('dow' :: text, ga.webpublicationdate)                        AS day_of_week,
    date_part('doy' :: text, ga.webpublicationdate)                        AS day_of_year,
    date_part('hour' :: text, ga.webpublicationdate)                       AS hour,
    date_part('minute' :: text, ga.webpublicationdate)                     AS minute,
    CASE
    WHEN ((ar.rank) :: numeric <= wtp.ten_percent)
      THEN 'TRUE' :: text
    ELSE 'FALSE' :: text
    END                                                                    AS in_top_ten_percent,
    acc.comment_count,
    cv.score                                                               AS competitive_score
  FROM guardianapi ga,
    categories c,
    articlecommentcountview acc,
    articlecountrankview ar,
    weeklytenpercentview wtp,
    competitiveview cv
  WHERE ((ga.article_id = acc.article_id) AND (acc.article_id = ar.article_id) AND (acc.week = wtp.week) AND
         (acc.year = wtp.year) AND (ga.sectionid = c.category) AND (cv.article_id = ga.article_id) AND
         (NOT ((acc.comment_count = 50) OR
               ((date_part('day' :: text, ga.webpublicationdate) = (29) :: double precision) AND
                (date_part('month' :: text, ga.webpublicationdate) = (2) :: double precision)))))
  GROUP BY ga.article_id, ga.headline, ga.bodytext, c.id, acc.comment_count, ar.rank, wtp.ten_percent, cv.score;

create function competitive_score(diff double precision, va double precision [], vb double precision [], dim integer)
  returns double precision
language plpgsql
as $$
DECLARE
  di double precision;
  t  double precision;
BEGIN
  t := sigmoid_derivative(diff);
  di := distance(va, vb, dim);

  RETURN t / power(di, 2);
END;
$$;

create function distance(l double precision [], r double precision [], dim integer)
  returns double precision
language plpgsql
as $$
DECLARE
  s double precision;
BEGIN
  s := 0;
  FOR i IN 1..dim LOOP
    s := s + ((l [i] - r [i]) * (l [i] - r [i]));
  END LOOP;

  RETURN |/s;
END;
$$;

create function sigmoid(x double precision)
  returns double precision
language plpgsql
as $$
DECLARE
  e double precision;
BEGIN
  e := exp(1.00);

  RETURN 1 / (1 + power(e, -x));
END;
$$;

create function sigmoid_derivative(x double precision)
  returns double precision
language plpgsql
as $$
DECLARE
  s double precision;
BEGIN
  s := sigmoid(x);

  RETURN s * (1 - s);
END;
$$;
